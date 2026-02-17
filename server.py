#!/usr/bin/env python3
"""
CSM Streaming TTS Server
Provides WebSocket-based real-time audio streaming
"""

import os
os.environ["NO_CUDA_GRAPH"] = "1"

import asyncio
import torch
import numpy as np
import wave
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import json
from pathlib import Path

# Add current dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_loader import load_csm_1b
from common import Segment
from main_streaming import prepare_prompt, SPEAKER_PROMPTS

app = FastAPI(title="CSM Streaming TTS Server")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
generator = None
prompt_context = None

def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio to WAV bytes"""
    if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
        audio_int = (audio_np * 32767).astype(np.int16)
    else:
        audio_int = audio_np.astype(np.int16)
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    return wav_buffer.getvalue()

@app.on_event("startup")
async def startup_event():
    """Load model on server startup"""
    global generator, prompt_context
    
    print("=" * 60)
    print("Starting CSM Streaming TTS Server")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading CSM-1B model (this may take a moment)...")
    generator = load_csm_1b(device)
    
    # Prepare prompt context
    print("Loading speaker prompt...")
    prompt_context = [prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.audio_decoder.sample_rate 
    )]
    
    # Warmup
    print("Running warmup...")
    try:
        for _ in generator.generate_stream(
            text="Warmup.",
            speaker=0,
            context=prompt_context,
            max_audio_length_ms=500,
            temperature=0.9,
            topk=50
        ):
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")
    
    print("=" * 60)
    print("Server ready!")
    print("=" * 60)

@app.get("/")
async def get_client():
    """Serve the web client"""
    client_path = Path(__file__).parent / "client.html"
    if client_path.exists():
        return FileResponse(client_path)
    else:
        return HTMLResponse("""
        <html>
            <head><title>CSM TTS Client</title></head>
            <body>
                <h1>Error: client.html not found</h1>
                <p>Please create client.html in the same directory as server.py</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for streaming audio generation"""
    await websocket.accept()
    
    try:
        while True:
            # Receive text from client
            data = await websocket.receive_json()
            text = data.get("text", "")
            temperature = data.get("temperature", 0.9)
            topk = data.get("topk", 50)
            
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue
            
            print(f"Generating audio for: {text[:50]}...")
            
            # Send start signal
            await websocket.send_json({
                "type": "start",
                "text": text,
                "sample_rate": generator.audio_decoder.sample_rate
            })
            
            # Track metrics
            chunk_count = [0]
            start_time = asyncio.get_event_loop().time()
            first_chunk_time = [None]
            
            try:
                print(f"[Server] Starting generation...")
                
                # Iterate through generator, yielding control to event loop
                for audio_chunk in generator.generate_stream(
                    text=text,
                    speaker=0,
                    context=prompt_context,
                    max_audio_length_ms=120_000,
                    temperature=temperature,
                    topk=topk
                ):
                    chunk_count[0] += 1
                    current_time = asyncio.get_event_loop().time()
                    
                    if first_chunk_time[0] is None:
                        first_chunk_time[0] = current_time - start_time
                        print(f"First chunk latency: {first_chunk_time[0]*1000:.1f}ms")
                    
                    # Convert to numpy
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy() if audio_chunk.is_cuda else audio_chunk.numpy()
                    else:
                        audio_np = audio_chunk
                    
                    # Log chunk details
                    chunk_duration = len(audio_np) / generator.audio_decoder.sample_rate
                    print(f"Chunk {chunk_count[0]}: {len(audio_np)} samples ({chunk_duration:.2f}s)")
                    
                    # Convert to WAV bytes
                    wav_bytes = audio_to_wav_bytes(audio_np, generator.audio_decoder.sample_rate)
                    
                    # Send immediately via WebSocket
                    import base64
                    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                    
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": audio_b64,
                        "chunk_id": chunk_count[0],
                        "sample_rate": generator.audio_decoder.sample_rate
                    })
                    print(f"[Server] Sent chunk {chunk_count[0]} to client")
                    
                    # Yield control to event loop and ensure send completes
                    await asyncio.sleep(0.001)  # 1ms to ensure network send happens
                
                print(f"[Server] Generation complete, sent {chunk_count[0]} chunks")
                
                # Send completion signal
                total_time = asyncio.get_event_loop().time() - start_time
                await websocket.send_json({
                    "type": "complete",
                    "chunks": chunk_count,
                    "total_time": total_time,
                    "first_chunk_latency": first_chunk_time
                })
                
                print(f"Generation complete: {chunk_count} chunks in {total_time:.2f}s")
                
            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSM Streaming TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8998, help="Port to bind to")
    args = parser.parse_args()
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"Open your browser to: http://localhost:{args.port}\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
