"""
Decodes arbitrary uploaded audio (mp3, m4a, ogg, webm/opus from browsers, wav, etc.)
into a mono float32 numpy array at the target sample rate, via ffmpeg.

Doing this ourselves (instead of trusting the upload's declared format) means the
API accepts literally anything ffmpeg can read, and we never hand a mystery
container straight to the model.
"""
import numpy as np
import subprocess
import shutil

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg not found on PATH — install it in the container image.")


def decode_audio(raw_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",                 # mono
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=raw_bytes, capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='ignore')[-500:]}")

    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def get_duration_seconds(audio: np.ndarray, sr: int = 16000) -> float:
    return len(audio) / sr
