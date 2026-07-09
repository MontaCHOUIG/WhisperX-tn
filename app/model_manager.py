"""
Owns the loaded model(s) as process-wide singletons and exposes one blocking
`transcribe()` call. Blocking is intentional: it's called from inside the GPU
worker thread (see gpu_worker.py), never directly from an event-loop coroutine.

Design notes specific to this deployment:
- `whisperx.load_model` accepts a local CTranslate2 model directory directly
  (it's a thin wrapper around faster-whisper's WhisperModel), so your finetuned
  model just needs to be a valid ct2-transformers-converter output directory.
- Language is pinned via settings.LANGUAGE instead of auto-detected. Whisper's
  language-ID head was trained on standard language codes and is unreliable
  for dialectal Arabic — it's a coin flip whether it lands on "ar", and if you
  finetuned specifically for Tunisian audio you already know what's in it.
- Alignment (wav2vec2 forced alignment) is off by default. There is no
  production-grade Tunisian-derja phoneme model; the closest public option is
  an MSA wav2vec2-CTC model, which will misalign dialectal audio in
  systematic ways (different vowel elision, French/Berber loanwords, etc).
  If you turn it on, treat the resulting word timestamps as approximate.
  Segment-level timestamps from the CT2 model itself are unaffected either way.
"""
import logging
import numpy as np
import whisperx
from whisperx.diarize import DiarizationPipeline

from .config import settings
from .schemas import TranscriptionResult, Segment, Word

logger = logging.getLogger("model_manager")


class ModelManager:
    def __init__(self):
        self._asr_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None

    def load(self):
        logger.info(f"Loading ASR model from {settings.MODEL_PATH} "
                    f"(device={settings.DEVICE}, compute_type={settings.COMPUTE_TYPE})")
        self._asr_model = whisperx.load_model(
            settings.MODEL_PATH,
            device=settings.DEVICE,
            device_index=settings.DEVICE_INDEX,
            compute_type=settings.COMPUTE_TYPE,
            language=settings.LANGUAGE,
        )
        logger.info("ASR model loaded.")

        if settings.ENABLE_ALIGNMENT:
            align_lang = settings.LANGUAGE
            logger.info(f"Loading alignment model for language={align_lang} "
                        f"(model_name={settings.ALIGN_MODEL_NAME or 'default'})")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=align_lang,
                device=settings.DEVICE,
                model_name=settings.ALIGN_MODEL_NAME,
            )

        if settings.ENABLE_DIARIZATION:
            logger.info("Loading diarization pipeline...")
            self._diarize_model = DiarizationPipeline(
                use_auth_token=settings.HF_TOKEN,
                device=settings.DEVICE,
            )

    def is_ready(self) -> bool:
        return self._asr_model is not None

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = None,
        beam_size: int = None,
        batch_size: int = None,
        align: bool = False,
        diarize: bool = False,
        min_speakers: int = None,
        max_speakers: int = None,
        initial_prompt: str = None,
    ) -> TranscriptionResult:
        language = language or settings.LANGUAGE
        batch_size = batch_size or settings.BATCH_SIZE

        result = self._asr_model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
        )

        if align:
            if self._align_model is None:
                raise RuntimeError(
                    "Alignment requested but ENABLE_ALIGNMENT=false on the server "
                    "(or no alignment model was loaded at startup)."
                )
            result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                settings.DEVICE,
                return_char_alignments=False,
            )

        if diarize:
            if self._diarize_model is None:
                raise RuntimeError(
                    "Diarization requested but ENABLE_DIARIZATION=false on the server."
                )
            diarize_segments = self._diarize_model(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._to_schema(result, language, len(audio) / settings.TARGET_SAMPLE_RATE)

    @staticmethod
    def _to_schema(raw: dict, language: str, duration: float) -> TranscriptionResult:
        segments = []
        full_text_parts = []
        for i, seg in enumerate(raw["segments"]):
            words = None
            if seg.get("words"):
                words = [
                    Word(
                        word=w.get("word", ""),
                        start=w.get("start"),
                        end=w.get("end"),
                        score=w.get("score"),
                        speaker=w.get("speaker"),
                    )
                    for w in seg["words"]
                ]
            segments.append(
                Segment(
                    id=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    words=words,
                    speaker=seg.get("speaker"),
                    avg_logprob=seg.get("avg_logprob"),
                    no_speech_prob=seg.get("no_speech_prob"),
                )
            )
            full_text_parts.append(seg["text"].strip())

        return TranscriptionResult(
            language=language,
            duration_seconds=duration,
            segments=segments,
            text=" ".join(full_text_parts).strip(),
        )


model_manager = ModelManager()
