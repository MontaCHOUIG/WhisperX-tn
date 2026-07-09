"""
Why a manual queue instead of just letting FastAPI/uvicorn handle concurrency:

- CTranslate2 + pyannote calls are synchronous and GPU-bound. If N requests hit
  the event loop at once and each just gets thrown at a thread pool, they all
  land on the GPU simultaneously -> VRAM spikes, unpredictable latency, and on
  Blackwell in particular, less-tested concurrent-stream behavior than on
  Ampere/Hopper.
- A single background worker consuming an asyncio.Queue gives you a
  predictable, ordered, back-pressured pipeline: one job on the GPU at a time
  (or up to MAX_CONCURRENT_GPU_JOBS if you've validated your VRAM headroom for
  running more), everything else waits in queue with a visible position.
- The blocking model call itself still runs in a thread executor so it doesn't
  freeze the event loop for other requests (health checks, job polling, etc).

If you outgrow one process, don't add more workers to *this* queue — instead
run multiple uvicorn processes (one per GPU, or one per model replica if VRAM
allows) behind a load balancer, each with its own queue, and move job state
into Redis so any process can serve a status poll.
"""
import asyncio
import logging
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import settings
from .model_manager import model_manager
from .schemas import JobStatus

logger = logging.getLogger("gpu_worker")


@dataclass
class Job:
    id: str
    kwargs: dict
    status: JobStatus = JobStatus.queued
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class GPUWorker:
    def __init__(self, max_concurrent: int = 1):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=settings.GPU_QUEUE_MAXSIZE)
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._workers_started = False

    def start(self, n_consumers: int = 1):
        if self._workers_started:
            return
        for _ in range(n_consumers):
            asyncio.create_task(self._consume())
        self._workers_started = True

    async def _consume(self):
        loop = asyncio.get_event_loop()
        while True:
            job: Job = await self._queue.get()
            async with self._semaphore:
                job.status = JobStatus.processing
                try:
                    result = await loop.run_in_executor(
                        self._executor, self._run_blocking, job.kwargs
                    )
                    job.result = result
                    job.status = JobStatus.done
                except Exception as e:
                    logger.exception(f"Job {job.id} failed")
                    job.error = str(e)
                    job.status = JobStatus.error
                finally:
                    self._queue.task_done()

    @staticmethod
    def _run_blocking(kwargs: dict):
        return model_manager.transcribe(**kwargs)

    async def submit(self, **kwargs) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, kwargs=kwargs)
        self._jobs[job_id] = job
        await self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> Optional[int]:
        # approximate: position among still-queued jobs, ordered by submission time
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.queued:
            return None
        queued = sorted(
            (j for j in self._jobs.values() if j.status == JobStatus.queued),
            key=lambda j: j.created_at,
        )
        return queued.index(job)

    async def run_sync(self, **kwargs):
        """For short audio: bypass job bookkeeping but still go through the same
        semaphore + executor so it can't jump ahead of the GPU's actual capacity."""
        loop = asyncio.get_event_loop()
        async with self._semaphore:
            return await loop.run_in_executor(self._executor, self._run_blocking, kwargs)


gpu_worker = GPUWorker(max_concurrent=settings.MAX_CONCURRENT_GPU_JOBS)
