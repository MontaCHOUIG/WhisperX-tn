import logging
import json

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
    status,
)

from .config import settings
from .schemas import (
    TranscriptionResult, JobSubmitResponse, JobStatusResponse, JobStatus,
)
from .audio_utils import decode_audio, get_duration_seconds
from .model_manager import model_manager
from .gpu_worker import gpu_worker
from .auth import require_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="Tunisian WhisperX ASR Backend", version="1.0.0")


@app.on_event("startup")
async def startup():
    model_manager.load()
    gpu_worker.start(n_consumers=1)  # 1 consumer regardless of MAX_CONCURRENT_GPU_JOBS;
                                      # the semaphore inside it caps actual GPU concurrency


@app.get("/v1/health")
async def health():
    return {
        "status": "ok" if model_manager.is_ready() else "loading",
        "device": settings.DEVICE,
        "compute_type": settings.COMPUTE_TYPE,
        "model_path": settings.MODEL_PATH,
    }


def _read_upload_and_decode(raw: bytes) -> tuple:
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.MAX_UPLOAD_MB} MB limit.")
    try:
        audio = decode_audio(raw, target_sr=settings.TARGET_SAMPLE_RATE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    duration = get_duration_seconds(audio, settings.TARGET_SAMPLE_RATE)
    return audio, duration


@app.post(
    "/v1/transcribe",
    response_model=TranscriptionResult,
    dependencies=[Depends(require_api_key)],
)
async def transcribe_sync(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    beam_size: int = Form(default=None),
    batch_size: int = Form(default=None),
    diarize: bool = Form(default=False),
    align: bool = Form(default=False),
    min_speakers: int = Form(default=None),
    max_speakers: int = Form(default=None),
    initial_prompt: str = Form(default=None),
):
    """
    Synchronous path: blocks until the transcript is ready. Only use this for
    short clips — anything at or above SYNC_MAX_AUDIO_SECONDS is rejected with
    a 413 pointing the caller at /v1/transcribe/async instead, since holding an
    HTTP connection open for minutes is fragile (proxy timeouts, client retries
    stacking up more GPU work behind the scenes, etc).
    """
    raw = await file.read()
    audio, duration = _read_upload_and_decode(raw)

    if duration > settings.SYNC_MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio is {duration:.1f}s, longer than the {settings.SYNC_MAX_AUDIO_SECONDS}s "
                "sync limit. Use POST /v1/transcribe/async instead."
            ),
        )

    result = await gpu_worker.run_sync(
        audio=audio,
        language=language,
        beam_size=beam_size,
        batch_size=batch_size,
        align=align,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        initial_prompt=initial_prompt,
    )
    return result


@app.post(
    "/v1/transcribe/async",
    response_model=JobSubmitResponse,
    dependencies=[Depends(require_api_key)],
)
async def transcribe_async(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    beam_size: int = Form(default=None),
    batch_size: int = Form(default=None),
    diarize: bool = Form(default=False),
    align: bool = Form(default=False),
    min_speakers: int = Form(default=None),
    max_speakers: int = Form(default=None),
    initial_prompt: str = Form(default=None),
):
    """Submit a job, get a job_id back immediately, poll /v1/jobs/{job_id}."""
    raw = await file.read()
    audio, _duration = _read_upload_and_decode(raw)

    job = await gpu_worker.submit(
        audio=audio,
        language=language,
        beam_size=beam_size,
        batch_size=batch_size,
        align=align,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        initial_prompt=initial_prompt,
    )
    return JobSubmitResponse(
        job_id=job.id,
        status=job.status,
        poll_url=f"/v1/jobs/{job.id}",
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_job(job_id: str):
    job = gpu_worker.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        result=job.result,
        error=job.error,
        queue_position=gpu_worker.queue_position(job_id),
    )


def _websocket_api_key_is_valid(websocket: WebSocket, api_key: str = None) -> bool:
    if settings.API_KEY is None:
        return True
    return api_key == settings.API_KEY or websocket.headers.get("x-api-key") == settings.API_KEY


def _pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
    if len(raw) % 2 != 0:
        raise ValueError("PCM frames must be 16-bit signed little-endian samples.")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@app.websocket("/v1/transcribe/stream")
async def transcribe_stream(
    websocket: WebSocket,
    api_key: str = Query(default=None),
    language: str = Query(default=None),
    emit_every_seconds: float = Query(default=None, gt=0),
    window_seconds: float = Query(default=None, gt=0),
):
    """
    Live-ish transcription over WebSocket.

    The client sends binary frames containing raw mono pcm_s16le audio at
    TARGET_SAMPLE_RATE. The server transcribes a rolling window every few
    seconds and sends JSON partials back. Send {"event":"stop"} as text to
    request one final transcript for the buffered audio.
    """
    if not _websocket_api_key_is_valid(websocket, api_key):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    await websocket.send_json({
        "event": "ready",
        "sample_rate": settings.TARGET_SAMPLE_RATE,
        "encoding": "pcm_s16le",
        "channels": 1,
    })

    emit_every = emit_every_seconds or settings.STREAM_TRANSCRIBE_EVERY_SECONDS
    window_size = window_seconds or settings.STREAM_WINDOW_SECONDS
    emit_every_samples = int(emit_every * settings.TARGET_SAMPLE_RATE)
    window_samples = int(window_size * settings.TARGET_SAMPLE_RATE)

    audio = np.empty(0, dtype=np.float32)
    samples_since_emit = 0
    total_samples_received = 0

    async def send_transcript(event: str, source_audio: np.ndarray, offset_samples: int):
        if source_audio.size == 0:
            return
        result = await gpu_worker.run_sync(
            audio=source_audio,
            language=language,
            align=False,
            diarize=False,
        )
        offset_seconds = offset_samples / settings.TARGET_SAMPLE_RATE
        await websocket.send_json({
            "event": event,
            "text": result.text,
            "window_start_seconds": offset_seconds,
            "window_duration_seconds": result.duration_seconds,
            "segments": [
                {
                    "start": segment.start + offset_seconds,
                    "end": segment.end + offset_seconds,
                    "text": segment.text,
                }
                for segment in result.segments
            ],
        })

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                chunk = _pcm16_bytes_to_float32(message["bytes"])
                audio = np.concatenate([audio, chunk])
                samples_since_emit += chunk.size
                total_samples_received += chunk.size

                if audio.size > window_samples:
                    audio = audio[-window_samples:]

                if samples_since_emit >= emit_every_samples:
                    samples_since_emit = 0
                    offset = max(0, total_samples_received - audio.size)
                    await send_transcript("partial", audio, offset)

            elif "text" in message and message["text"] is not None:
                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "error": "Text frames must be JSON commands."})
                    continue

                if command.get("event") == "stop":
                    offset = max(0, total_samples_received - audio.size)
                    await send_transcript("final", audio, offset)
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                    return
                await websocket.send_json({"event": "error", "error": "Unsupported command."})

            elif message.get("type") == "websocket.disconnect":
                return

    except WebSocketDisconnect:
        logger.info("Streaming transcription websocket disconnected.")
    except ValueError as e:
        await websocket.send_json({"event": "error", "error": str(e)})
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
