# Tunisian WhisperX ASR Backend

FastAPI service wrapping a custom finetuned, CT2-converted Whisper model
(WhisperX-style pipeline: VAD-batched decoding, optional forced alignment,
optional diarization) for Tunisian dialect ASR, targeting an RTX 5090 (Blackwell).

## 1. Preparing your model

Your finetuned checkpoint needs to already be in `ct2-transformers-converter`
output format. If you haven't converted it yet:

```bash
pip install ctranslate2 transformers
ct2-transformers-converter \
  --model /path/to/your/finetuned-whisper-hf-checkpoint \
  --output_dir ./models/whisper-tn-ct2 \
  --quantization float16 \
  --copy_files tokenizer.json preprocessor_config.json
```

Mount `./models/whisper-tn-ct2` into the container at `/models/whisper-tn-ct2`
(already wired up in `docker-compose.yml`).

## 2. Hardware-specific gotchas (read before deploying)

- **int8 on sm_120**: some CTranslate2 builds crash with
  `CUBLAS_STATUS_NOT_SUPPORTED` when running int8 on RTX 50-series cards — a
  padding bug in the INT8 tensor core path. Default here is `float16`. If you
  want the speed/memory win of int8, pin CTranslate2 to a version that
  includes the "multiple of 16 padding" fix and validate it yourself against
  your model before trusting it in production.
- **PyTorch build**: `pyannote.audio` (used for diarization and, depending on
  version, VAD) needs a PyTorch build that actually ships sm_120 kernels.
  Stable PyTorch releases lag new hardware — check
  https://pytorch.org/get-started/locally/ for the current recommended
  cu128/cu129 index before you build the image; you may need a recent nightly.
- **cuDNN 9**: recent CTranslate2 releases dropped cuDNN 8 support. Use a
  cuDNN-9-based CUDA image (the provided Dockerfile does).
- **Forced alignment**: there's no solid Tunisian-derja wav2vec2 phoneme
  model. `ENABLE_ALIGNMENT` defaults to `false`; segment-level timestamps
  from the Whisper model itself are unaffected. If you turn alignment on
  with an MSA Arabic model, expect word-level timing drift on dialectal audio.
- **Diarization** requires an `HF_TOKEN` and accepting pyannote's model terms
  on Hugging Face (`pyannote/segmentation-3.0`,
  `pyannote/speaker-diarization-3.1` or whatever version you pin) — otherwise
  you get a 401 that has nothing to do with your own API key.

## 3. Running it

```bash
cp .env.example .env      # edit MODEL_PATH, API_KEY, etc.
docker compose up --build
curl http://localhost:8000/v1/health
```

## 4. API reference

All endpoints except `/v1/health` require an `X-API-Key` header if `API_KEY`
is set server-side.

### `POST /v1/transcribe` — synchronous

For short clips only (default limit: 60s — see `SYNC_MAX_AUDIO_SECONDS`).
Blocks until the transcript is ready; returns it directly.

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -H "X-API-Key: change-me" \
  -F "file=@clip.wav" \
  -F "language=ar" \
  -F "diarize=false" \
  -F "align=false"
```

Response (`200`):
```json
{
  "language": "ar",
  "duration_seconds": 12.4,
  "text": "...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "...",
      "words": null,
      "speaker": null,
      "avg_logprob": -0.18,
      "no_speech_prob": 0.01
    }
  ]
}
```

Errors: `400` (undecodable audio), `413` (over size limit, or audio too long
for the sync path — the message tells you to use the async endpoint), `422`
(bad form data), `503` (model still loading).

### `POST /v1/transcribe/async` — job-based, for anything longer

Returns immediately with a `job_id`; you poll for the result. Use this for
anything you wouldn't want to hold an HTTP connection open for — meeting
recordings, batch files, phone call archives.

```bash
curl -X POST http://localhost:8000/v1/transcribe/async \
  -H "X-API-Key: change-me" \
  -F "file=@long_meeting.mp3" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

```json
{ "job_id": "9f1c...", "status": "queued", "poll_url": "/v1/jobs/9f1c..." }
```

### `GET /v1/jobs/{job_id}` — poll

```bash
curl http://localhost:8000/v1/jobs/9f1c... -H "X-API-Key: change-me"
```

```json
{
  "job_id": "9f1c...",
  "status": "processing",
  "result": null,
  "error": null,
  "queue_position": null
}
```

`status` is one of `queued`, `processing`, `done`, `error`. When `done`,
`result` has the same shape as the sync endpoint's response. Poll on a
backoff (e.g. 1s → 2s → 5s) rather than tight-looping — there's no websocket
push for job completion in this version.

### `WebSocket /v1/transcribe/stream` — continuous microphone-style input

Use this when a client starts recording and wants transcript updates while
audio is still arriving. This is a rolling-window stream, not token-by-token
Whisper decoding: the client sends small raw PCM chunks, and the server
re-transcribes the latest window every few seconds.

Connect with either an `X-API-Key` websocket header or an `api_key` query
parameter:

```text
ws://localhost:8000/v1/transcribe/stream?api_key=change-me
```

Query options:

- `language=ar` overrides the server default language.
- `emit_every_seconds=3` controls how often partial transcripts are emitted.
- `window_seconds=12` controls how much recent audio each partial uses.

Audio contract: send binary websocket frames as mono `pcm_s16le` at
`16000 Hz`. The server replies with JSON:

```json
{
  "event": "partial",
  "text": "...",
  "window_start_seconds": 6.0,
  "window_duration_seconds": 12.0,
  "segments": [{ "start": 6.2, "end": 8.9, "text": "..." }]
}
```

Send a text frame to stop:

```json
{ "event": "stop" }
```

Minimal browser-side shape:

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/transcribe/stream?api_key=change-me");
ws.binaryType = "arraybuffer";
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.event === "partial" || msg.event === "final") {
    transcriptElement.textContent = msg.text;
  }
};

// Feed ws.send(...) with 16 kHz mono Int16Array PCM chunks from an AudioWorklet.
// On stop:
ws.send(JSON.stringify({ event: "stop" }));
```

For browser microphones, capture audio with Web Audio, downsample to 16 kHz
mono, convert Float32 samples to signed 16-bit PCM, then send each chunk's
`ArrayBuffer`.

### `GET /v1/health`

```bash
curl http://localhost:8000/v1/health
```
```json
{"status": "ok", "device": "cuda", "compute_type": "float16", "model_path": "/models/whisper-tn-ct2"}
```

## 5. Client examples

### Python (sync, short clip)

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/transcribe",
    headers={"X-API-Key": "change-me"},
    files={"file": open("clip.wav", "rb")},
    data={"language": "ar"},
)
resp.raise_for_status()
result = resp.json()
print(result["text"])
```

### Python (async, long file, with polling)

```python
import requests
import time

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "change-me"}

def transcribe_long(path, **options):
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/v1/transcribe/async",
            headers=HEADERS,
            files={"file": f},
            data=options,
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    delay = 1.0
    while True:
        time.sleep(delay)
        r = requests.get(f"{BASE}/v1/jobs/{job_id}", headers=HEADERS)
        r.raise_for_status()
        job = r.json()
        if job["status"] == "done":
            return job["result"]
        if job["status"] == "error":
            raise RuntimeError(job["error"])
        delay = min(delay * 1.5, 5.0)

result = transcribe_long("long_meeting.mp3", diarize="true", min_speakers="2")
for seg in result["segments"]:
    speaker = seg.get("speaker", "?")
    print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {speaker}: {seg['text']}")
```

### JavaScript / fetch (browser or Node 18+)

```javascript
async function transcribe(file, apiKey) {
  const form = new FormData();
  form.append("file", file);
  form.append("language", "ar");

  const res = await fetch("http://localhost:8000/v1/transcribe", {
    method: "POST",
    headers: { "X-API-Key": apiKey },
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

### cURL one-liner health/smoke test

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/health
```

## 6. Scaling beyond one process

The in-memory job store and single GPU queue in this project are correct for
**one process per GPU**. If you need more throughput:

- Run one container per physical GPU (never multiple `uvicorn` workers
  sharing one GPU — they'd each load a full copy of the model into VRAM).
- Move job state from the in-memory dict (`gpu_worker.py`) into Redis so any
  process can answer a status poll, and put a load balancer in front.
- If you have VRAM headroom (RTX 5090 has 32GB), you can raise
  `MAX_CONCURRENT_GPU_JOBS` above 1 — but benchmark it: CTranslate2's own
  batching (`BATCH_SIZE`) is usually a better lever than running multiple
  concurrent Python-level calls into the same GPU.
