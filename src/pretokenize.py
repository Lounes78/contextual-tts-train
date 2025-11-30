"""
Efficient pre-tokenization script using chunked Parquet files.
Updated to work with peoples_speech Parquet dataset format.
"""
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import glob
import io
import atexit
import gc
from datasets import load_dataset, Audio

import torch._dynamo
# This forces PyTorch to ignore the compiler error and run normally (Eager mode)
torch._dynamo.config.suppress_errors = True

# Handle both direct execution and module import
try:
    from . import config
    from .utils import load_tokenizers
except ImportError:
    import config
    from utils import load_tokenizers


def cleanup_gpu():
    """
    Force garbage collection and clear CUDA cache on script exit.
    This helps prevent 'Zombie' processes holding GPU memory.
    """
    if torch.cuda.is_available():
        print("\nCleaning up GPU memory...")
        # Force Python's garbage collector to release unused variables
        gc.collect()
        # Force PyTorch to release cached memory back to the OS
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU memory released.")

# Register this function to run automatically when the script ends
atexit.register(cleanup_gpu)


def load_parquet_dataset(parquet_data_path: str, split: str):
    """
    Load dataset from Parquet files using datasets library.
    Handles the peoples_speech dataset structure.
    """
    data_path = Path(parquet_data_path)
    
    if split == "train":
        pattern = "train-*.parquet"
    elif split == "val" or split == "validation":
        pattern = "validation-*.parquet"
    elif split == "test":
        pattern = "test-*.parquet"
    else:
        raise ValueError(f"Unsupported split: {split}. Use 'train', 'validation', or 'test'")
    
    # Check if any files exist for this split
    matching_files = list(data_path.glob(pattern))
    if not matching_files:
        return None  # No files found for this split
    
    try:
        # Use datasets library to load with streaming for memory efficiency
        ds = load_dataset(
            "parquet",
            data_files=str(data_path / pattern),
            split="train",  # datasets always uses "train" for the split name
            streaming=True
        )
        return ds
    except Exception as e:
        print(f"Failed to load {split} split: {e}")
        return None


def get_next_chunk_number(output_dir, split):
    """Get the next chunk number by checking existing chunk files."""
    chunk_pattern = f"{split}_part_*.parquet"
    existing_chunks = glob.glob(str(Path(output_dir) / chunk_pattern))
    
    if not existing_chunks:
        return 1
    
    chunk_numbers = []
    for chunk_file in existing_chunks:
        filename = Path(chunk_file).stem
        try:
            chunk_num = int(filename.split('_')[-1])
            chunk_numbers.append(chunk_num)
        except ValueError:
            continue
    
    return max(chunk_numbers) + 1 if chunk_numbers else 1


def get_num_existing_samples(output_dir, split):
    """
    Count total samples across all existing chunk files.
    OPTIMIZATION: Only reads metadata, not full data (O(1) vs O(N)).
    """
    chunk_pattern = f"{split}_part_*.parquet"
    existing_chunks = glob.glob(str(Path(output_dir) / chunk_pattern))
    
    total_samples = 0
    for chunk_file in existing_chunks:
        try:
            # CRITICAL OPTIMIZATION: Only load metadata, not full data
            df = pd.read_parquet(chunk_file, columns=[])
            total_samples += len(df)
        except Exception:
            continue
    
    return total_samples


def write_chunk(output_dir, split, chunk_num, audio_tokens_batch, text_tokens_batch):
    """Write a single chunk file."""
    chunk_filename = f"{split}_part_{chunk_num:03d}.parquet"
    chunk_path = Path(output_dir) / chunk_filename
    
    # Prepare data for this chunk
    chunk_data = []
    for i in range(len(audio_tokens_batch)):
        # Flatten audio array (matching original logic)
        audio_array = np.array(audio_tokens_batch[i], dtype=np.int32).flatten()
        text_array = np.array(text_tokens_batch[i], dtype=np.int32)
        
        # Calculate sequence length and total length
        seq_len = audio_array.shape[0] // config.AUDIO_NUM_CODEBOOKS
        total_len = seq_len + len(text_array) + 1  # +1 for EOS frame
        
        chunk_data.append({
            'audio': audio_array.tolist(),
            'text': text_array.tolist(),
            'length': total_len
        })
    
    # Write chunk directly to new file
    df = pd.DataFrame(chunk_data)
    df.to_parquet(chunk_path, index=False)
    
    return chunk_path, len(chunk_data)


def process_audio_entry(audio_data, target_sr=24000):
    """
    Robustly handle audio data whether it's raw bytes OR already decoded.
    """
    try:
        # CASE 1: Audio is already decoded (Numpy Array) by datasets library
        if 'array' in audio_data:
            waveform = torch.from_numpy(audio_data['array']).float()
            sr = audio_data['sampling_rate']
            
            # Ensure it has a channel dimension [C, T]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                
        # CASE 2: Audio is raw bytes (dictionary contains 'bytes')
        elif 'bytes' in audio_data:
            audio_io = io.BytesIO(audio_data['bytes'])
            waveform, sr = torchaudio.load(audio_io)
            
        else:
            return None, None

        # --- Standardize Audio (Stereo -> Mono, Resample) ---
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
            
        return waveform, target_sr
        
    except Exception as e:
        print(f"Error processing audio entry: {e}")
        return None, None


def tokenize_and_store(parquet_data_path, output_dir, split, audio_tokenizer, text_tokenizer, device, save_every=1000):
    """
    Tokenize the dataset from Parquet format and save in chunked files.
    """
    # Load dataset using datasets library
    print(f"Loading {split} dataset from {parquet_data_path}...")
    ds = load_parquet_dataset(parquet_data_path, split)
    
    # Handle missing splits gracefully
    if ds is None:
        print(f"No data found for {split} split, skipping...")
        return
    
    # Check for existing samples and get starting point
    n_existing = get_num_existing_samples(output_dir, split)
    chunk_num = get_next_chunk_number(output_dir, split)
    
    if n_existing > 0:
        print(f"Resuming {split}: skipping {n_existing} already processed samples")
    else:
        print(f"Processing {split} split...")

    audio_tokens_batch, text_tokens_batch = [], []
    total_processed = 0
    skipped_count = 0

    # Process each sample
    pbar = tqdm(desc=f"Processing {split}", unit="samples")
    
    for example in ds:
        total_processed += 1
        
        # Skip samples if resuming
        if total_processed <= n_existing:
            skipped_count += 1
            if skipped_count % 1000 == 0:
                pbar.set_postfix({"skipped": skipped_count})
                pbar.update(1000)
            continue
        
        try:
            # Extract data from the example
            text = example['text'].strip()
            audio_data = example['audio']  # Get the whole dict
            
            # Basic validation
            if len(text) < 10:  # Skip very short texts
                continue
            
            # Process audio (Handles both Bytes and Decoded Arrays)
            waveform, sr = process_audio_entry(audio_data, config.MIMI_SAMPLE_RATE)
            
            if waveform is None:
                continue
                
            # Prepare audio for tokenizer (add batch and channel dimensions)
            waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]

            # Tokenize audio
            audio_tokens = audio_tokenizer.encode(waveform)[0].tolist()  # [n_codebooks, seq_len]
            
            # Get speaker ID (use a default since peoples_speech doesn't have speaker info)
            speaker = example.get('speaker', config.GEN_SPEAKER)
            
            # Tokenize text with speaker ID
            formatted_text = f"[{speaker}]{text}" if not config.OMIT_SPEAKER_ID else text
            text_tokens = text_tokenizer.encode(formatted_text)

            # Accumulate batch
            audio_tokens_batch.append(audio_tokens)
            text_tokens_batch.append(text_tokens)

            # Write chunk when batch is full
            if len(audio_tokens_batch) >= save_every:
                chunk_path, chunk_size = write_chunk(output_dir, split, chunk_num, audio_tokens_batch, text_tokens_batch)
                print(f"\nWrote chunk {chunk_num}: {chunk_path} ({chunk_size} samples)")
                
                chunk_num += 1
                audio_tokens_batch, text_tokens_batch = [], []
                
        except Exception as e:
            print(f"Error processing sample {total_processed}: {e}")
            continue
        
        # Update progress
        processed_new = total_processed - n_existing
        pbar.set_postfix({
            "processed": processed_new,
            "batch_size": len(audio_tokens_batch)
        })
        pbar.update(1)

    # Write final chunk if any data remains
    if audio_tokens_batch:
        chunk_path, chunk_size = write_chunk(output_dir, split, chunk_num, audio_tokens_batch, text_tokens_batch)
        print(f"\nWrote final chunk {chunk_num}: {chunk_path} ({chunk_size} samples)")

    pbar.close()
    print(f"Processed {total_processed - n_existing} new samples for {split}")


def main():
    """Main pretokenization function using configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizers
    print("Loading tokenizers...")
    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    # Ensure output directory exists
    Path(config.TOKENIZED_DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Process training data
    tokenize_and_store(
        config.PARQUET_DATA_PATH,
        output_dir=config.TOKENIZED_DATA_PATH,
        split="train",
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        device=device,
        save_every=config.SAVE_BATCH_SIZE
    )

    # Process validation data (if it exists)
    tokenize_and_store(
        config.PARQUET_DATA_PATH,
        output_dir=config.TOKENIZED_DATA_PATH,
        split="validation",
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        device=device,
        save_every=config.SAVE_BATCH_SIZE
    )

    # Process test data (if it exists)
    tokenize_and_store(
        config.PARQUET_DATA_PATH,
        output_dir=config.TOKENIZED_DATA_PATH,
        split="test",
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        device=device,
        save_every=config.SAVE_BATCH_SIZE
    )

    print(f"\nDone. Chunked tokenized data saved to: {config.TOKENIZED_DATA_PATH}")
    
    # Show what was created
    for split in ["train", "validation", "test"]:
        chunk_pattern = f"{split}_part_*.parquet"
        chunk_files = glob.glob(str(Path(config.TOKENIZED_DATA_PATH) / chunk_pattern))
        total_samples = get_num_existing_samples(config.TOKENIZED_DATA_PATH, split)
        print(f"{split}: {len(chunk_files)} chunks, {total_samples} total samples")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Pretokenization completed in {end_time - start_time:.2f} seconds.")