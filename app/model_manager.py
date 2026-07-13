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
import os
import threading
from pathlib import Path

import numpy as np

from .config import settings
from .schemas import TranscriptionResult, Segment, Word

# Import configuration first: it establishes HF_HOME before Hugging Face
# libraries are imported through WhisperX.
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger("model_manager")


class ModelManager:
    def __init__(self):
        self._asr_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None
        self._load_lock = threading.Lock()
        self._diarize_load_lock = threading.Lock()

    def load(self):
        """Load startup-critical models once per process.

        Diarization is deliberately excluded: it is optional per request and
        pyannote may fetch several Hub artifacts. `_get_diarize_model()` loads
        it once, only when a request enables diarization.
        """
        with self._load_lock:
            if self._asr_model is not None:
                logger.info("ASR model is already loaded; skipping startup reload.")
                return

            model_path = Path(
                settings.MODEL_PATH.replace("\\", os.sep)
            ).expanduser().resolve()
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"MODEL_PATH does not exist or is not a directory: {model_path}. "
                    "Set MODEL_PATH to the local CTranslate2 model directory; "
                    "use '/' separators on Linux."
                )
            if not (model_path / "model.bin").is_file():
                raise FileNotFoundError(
                    f"MODEL_PATH is missing model.bin: {model_path}. "
                    "Point it at the root of the CTranslate2-converted model."
                )

            logger.info(f"Loading ASR model from {model_path} "
                        f"(device={settings.DEVICE}, compute_type={settings.COMPUTE_TYPE})")
            self._asr_model = whisperx.load_model(
                str(model_path),
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

    def _get_diarize_model(self):
        """Return the process singleton, downloading Hub artifacts at most once."""
        if self._diarize_model is not None:
            return self._diarize_model

        with self._diarize_load_lock:
            if self._diarize_model is None:
                logger.info("Loading diarization pipeline on first diarized request...")
                self._diarize_model = DiarizationPipeline(
                    use_auth_token=settings.HF_TOKEN,
                    device=settings.DEVICE,
                )
                logger.info("Diarization pipeline loaded.")
        return self._diarize_model

    def is_ready(self) -> bool:
        return self._asr_model is not None

    def is_diarization_loaded(self) -> bool:
        return self._diarize_model is not None

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
            if not settings.ENABLE_DIARIZATION:
                raise RuntimeError(
                    "Diarization requested but ENABLE_DIARIZATION=false on the server."
                )
            diarize_segments = self._get_diarize_model()(
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
