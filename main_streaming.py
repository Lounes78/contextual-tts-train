import os
# Disable CUDA graphs — GH200 unified memory causes captures_underway assertion failures
# during F.conv1d in moshi's SEANet decoder. This is a hardware-level incompatibility.
os.environ["NO_CUDA_GRAPH"] = "1"

import argparse
import time
import torch
import queue
import threading
import platform
import wave
import numpy as np
from huggingface_hub import hf_hub_download
import torchaudio
import sys
from typing import List, Optional

# Add current dir to path to find local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add parent directory to path to import models if needed
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

try:
    from .common import Segment
    from .model_loader import load_csm_1b
    from .token_generator import TokenGenerator
except ImportError:
    # Fallback for running as script
    from common import Segment
    from model_loader import load_csm_1b
    from token_generator import TokenGenerator

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def stream_audio_to_wav(filename, sample_rate):
    """
    Initialize a WAV writer for streaming audio chunks.
    """
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    
    def write_chunk(audio_chunk):
        if isinstance(audio_chunk, torch.Tensor):
            audio_np = audio_chunk.detach().cpu().numpy()
        else:
            audio_np = audio_chunk
            
        if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
            audio_int = (audio_np * 32767).astype(np.int16)
        else:
            audio_int = audio_np.astype(np.int16)
            
        wav_file.writeframes(audio_int.tobytes())
    
    def close():
        wav_file.close()
        
    return write_chunk, close

def generate_streaming_audio(
    generator: TokenGenerator,
    text: str,
    speaker: int,
    context: List[Segment],
    output_file: str,
    max_audio_length_ms: float = 90_000,
    temperature: float = 1.0,
    topk: int = 50,
    play_audio: bool = False,
):
    """
    Generate audio with streaming output and timing metrics.
    """
    write_chunk, close_wav = stream_audio_to_wav(output_file, generator.audio_decoder.sample_rate)
    
    # Simple stats
    chunk_times = []
    chunk_count = 0
    total_audio_duration = 0
    start_time = time.time()
    latency_to_first_chunk = None

    def on_chunk_generated(chunk):
        nonlocal chunk_count, latency_to_first_chunk, total_audio_duration
        current_time = time.time()
        
        if chunk_count == 0:
            latency_to_first_chunk = current_time - start_time
            print(f"First chunk latency: {latency_to_first_chunk*1000:.1f}ms")
        
        write_chunk(chunk)
        
        chunk_count += 1
        chunk_duration = len(chunk) / generator.audio_decoder.sample_rate
        total_audio_duration += chunk_duration
        chunk_times.append(current_time)

    print(f"Starting generation for text: {text[:50]}...")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        for _ in generator.generate_stream(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            on_chunk_generated=on_chunk_generated
        ):
            pass # Consumption handled in on_chunk_generated
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
    close_wav()
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    # Calculate specialized metrics
    # If first chunk latency is huge (compilation), we calculate a "sustained RTF" 
    # that ignores the start-up time.
    
    sustained_elapsed = total_elapsed - (latency_to_first_chunk if latency_to_first_chunk else 0)
    # The first chunk itself also has duration, but usually small. 
    # If we ignore the time to get first chunk, we should roughly match sustained speed.
    first_chunk_duration = (len(chunk_times) > 0) and (chunk_times[0] - start_time) or 0
    # Actually simpler: just look at time between chunks
    
    if len(chunk_times) > 1:
        # Time from first chunk arrival to last chunk arrival
        generation_duration_after_first = chunk_times[-1] - chunk_times[0]
        # Audio generated after first chunk (approximate, assuming uniform chunks except maybe last)
        # But we track total_audio_duration. Let's make an approximation.
        # RTF = (Time taken) / (Audio duration)
        
        sustained_rtf = generation_duration_after_first / (total_audio_duration - (total_audio_duration/chunk_count)) if chunk_count > 1 else 0
    else:
        sustained_rtf = 0

    overall_rtf = total_elapsed / total_audio_duration if total_audio_duration > 0 else float('inf')
    
    print("\n" + "="*50)
    print("AUDIO GENERATION PERFORMANCE METRICS")
    print("="*50)
    if latency_to_first_chunk:
        print(f"First chunk latency: {latency_to_first_chunk*1000:.1f}ms")
    print(f"Total generation time: {total_elapsed:.2f}s")
    print(f"Audio duration: {total_audio_duration:.2f}s")
    print(f"Overall RTF (incl. compile/latency): {overall_rtf:.3f}x")
    if sustained_rtf > 0:
        print(f"Sustained RTF (excl. first chunk):   {sustained_rtf:.3f}x")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="CSM Streaming TTS")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the modular CSM streaming system.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    
    # Load model
    generator = load_csm_1b(args.device)
    
    # Prepare prompts first so we can use them in warmup for consistent shapes
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.audio_decoder.sample_rate 
    )

    # Warmup: run a short generation to trigger any lazy compilation/CUDA graph capture
    # so that the main generation timing is clean.
    print("\nRunning warmup (this may take a moment on first run)...")
    try:
        for _ in generator.generate_stream(
            text="Warmup.",
            speaker=0,
            context=[prompt_a],
            max_audio_length_ms=500,
            temperature=1.0,
            topk=50
        ):
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete. Starting main generation...\n")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")
        import traceback
        traceback.print_exc()

    # Generate
    generate_streaming_audio(
        generator=generator,
        text=args.text,
        speaker=0,
        context=[prompt_a],
        output_file=args.output
    )

if __name__ == "__main__":
    main()
