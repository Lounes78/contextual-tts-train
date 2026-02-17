# CSM-1B Streaming TTS

Real-time text-to-speech using CSM-1B with proper streaming.

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Open browser to `http://localhost:8998`

Type some text, click generate. Audio plays in real-time.

## Docker (Pre-configured Image)

A ready-to-use Docker image `csm-stream` is available with all dependencies and the environment already set up *on the gh200*

### Launch Command

```bash
docker run --gpus all -p 8998:8998 -v /home/az03732:/home/az03732 csm-stream
```


Once running, open your browser to `http://localhost:8998`.

## What Makes It Stream

- Generator yields chunks as they're ready (not waiting for completion)
- Server sends each chunk immediately via WebSocket
- Client plays chunks as they arrive using Web Audio API
- Progressive buffering: 15 → 25 → 40 frames

## Files

- `server.py` - FastAPI WebSocket server
- `client.html` - Browser client with Web Audio API
- `token_generator.py` - Streaming generator (yields chunks)
- `model_loader.py` - Loads CSM-1B model
- `decoding.py` - Audio decoder (tokens → audio)

- `CODE_WALKTHROUGH.txt` - Line-by-line execution trace from click to audio

## Performance (test on gh200)

- First chunk latency: 750ms-3s
- Chunk size: 1.2s → 2.0s (adaptive)
- RTF: 0.78-0.87x (faster than real-time)
- Sample rate: 24kHz, 16-bit PCM