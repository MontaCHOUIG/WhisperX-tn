from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Word(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None
    speaker: Optional[str] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


class TranscriptionOptions(BaseModel):
    language: Optional[str] = None          # override server default (e.g. force "ar")
    beam_size: Optional[int] = None
    batch_size: Optional[int] = None
    diarize: bool = False
    align: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    initial_prompt: Optional[str] = None    # useful to bias decoding toward Tunisian vocabulary/script conventions


class TranscriptionResult(BaseModel):
    language: str
    duration_seconds: float
    segments: List[Segment]
    text: str


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[TranscriptionResult] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None
