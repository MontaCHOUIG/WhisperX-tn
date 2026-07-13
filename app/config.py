"""
Central configuration. All values overridable via environment variables / .env.
"""
from typing import Optional
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Model ---
    MODEL_PATH: str = "models/whisper-tn-ct2"   # local dir with your CT2-converted finetuned model
    LANGUAGE: str = "ar"                          # force language (skip language-ID pass -> faster + safer for dialect)
    DEVICE: str = "cuda"
    DEVICE_INDEX: int = 0

    # sm_120 (RTX 5090) note: int8 GEMM has a known CTranslate2 padding bug on some builds.
    # Validate int8_float16 yourself before flipping this; float16 is the safe default.
    COMPUTE_TYPE: str = "float16"                 # float16 | int8_float16 | int8 | bfloat16

    BATCH_SIZE: int = 16                          # VAD-chunk batch size fed to the model at once
    BEAM_SIZE: int = 5

    # --- Alignment / diarization (optional stages) ---
    ENABLE_ALIGNMENT: bool = False               # no solid Tunisian-derja wav2vec2 model exists; see README
    ALIGN_MODEL_NAME: Optional[str] = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"     # (MSA, imperfect on derja)
    ENABLE_DIARIZATION: bool = True
    HF_TOKEN: Optional[str] = None                # required only if ENABLE_DIARIZATION or a gated align model
    # Store Hub models somewhere persistent. Mount this directory in containers;
    # otherwise pyannote has to download its models again after every replacement.
    HF_HOME: Optional[str] = ".cache/huggingface"

    # --- Concurrency / GPU serialization ---
    MAX_CONCURRENT_GPU_JOBS: int = 1              # keep at 1 unless you've validated multi-stream VRAM headroom
    GPU_QUEUE_MAXSIZE: int = 100

    # --- Sync vs async routing ---
    SYNC_MAX_AUDIO_SECONDS: int = 60              # requests longer than this are forced onto the async job path

    # --- Live streaming websocket ---
    STREAM_TRANSCRIBE_EVERY_SECONDS: float = 2.0  # emit a partial transcript after this much new audio
    STREAM_WINDOW_SECONDS: float = 12.0           # rolling audio window sent to Whisper for each partial

    JOB_TTL_SECONDS: int = 3600

    # --- Auth ---
    API_KEY: Optional[str] = None                 # if unset, auth is disabled (dev only)

    # --- Audio ---
    TARGET_SAMPLE_RATE: int = 16000
    MAX_UPLOAD_MB: int = 200

    class Config:
        env_file = ".env"


settings = Settings()

# huggingface_hub reads this setting while its modules are imported. Set it as
# soon as settings are available, before whisperx/pyannote is imported.
if settings.HF_HOME:
    os.environ.setdefault("HF_HOME", settings.HF_HOME)
