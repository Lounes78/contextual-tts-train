import os
import glob
import bisect
import polars as pl
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

AUDIO_NUM_CODEBOOKS = 32

def build_parquet_index(data_dir: str, split: str) -> Tuple[List[dict], List[int], List[int]]:
    """
    Scans directory for parquet chunks and builds a global index.
    
    Returns:
        file_infos: List of dicts with file path and start index.
        file_starts: List of start indices (for binary search).
        global_lengths: List of sequence lengths for every sample.
    """
    # Pattern matches: train_div0_part_001.parquet, train_part_001.parquet, etc.
    search_path = os.path.join(data_dir, f"{split}*part_*.parquet")
    files = sorted(glob.glob(search_path))
    if not files:
        print(f"Warning: No files found for split '{split}' in {data_dir}")
        return [], [], []

    print(f"Indexing {len(files)} parquet files for split '{split}'...")
    
    file_infos = []
    file_starts = []
    global_lengths = []
    current_start = 0

    for f_path in files:
        try:
            # We only need the 'length' column to build the index
            df = pl.scan_parquet(f_path).select("length").collect()
            n_rows = len(df)
            lengths = df["length"].to_list()
            
            file_infos.append({
                "path": f_path,
                "start": current_start,
                "len": n_rows
            })
            file_starts.append(current_start)
            
            global_lengths.extend(lengths)
            current_start += n_rows
        except Exception as e:
            print(f"Skipping corrupt file {f_path}: {e}")
        
    return file_infos, file_starts, global_lengths 




class ParquetTokenizedDataset(Dataset):
    """
    Polars-backed dataset for tokenized Parquet files.
    Supports efficient random access via lazy slicing or full in-memory loading.
    """

    def __init__(self, data_dir: str, split: str, load_in_memory: bool = False):
        self.data_dir = data_dir
        self.split = split
        self._in_memory = load_in_memory
        
        # 1. Build Index
        self.file_infos, self.file_starts, self._lengths = build_parquet_index(data_dir, split)
        self.total_len = len(self._lengths)

        # 2. Load into memory if requested (Fastest for training if RAM allows)
        if self._in_memory and self.total_len > 0:
            print(f"Loading {split} split into memory...")
            dfs = [pl.read_parquet(entry['path']) for entry in self.file_infos]
            if dfs:
                self.full_df = pl.concat(dfs)
            else:
                self.full_df = None
        else:
            self.full_df = None

    def __len__(self):
        return self.total_len

    def _get_file_info(self, idx: int):
        """Finds the specific file and local row index for a global index."""
        # Binary search to find the file index
        # bisect_right returns insertion point after idx, so subtract 1
        file_idx = bisect.bisect_right(self.file_starts, idx) - 1
        
        info = self.file_infos[file_idx]
        local_idx = idx - info['start']
        
        return info['path'], local_idx


    def __getitem__(self, idx: int):
        # A. Fetch Data
        if self._in_memory:
            # Polars fast row access
            row = self.full_df.row(idx, named=True)
            audio_list = row['audio']
            text_list = row['text']
        else:
            # Disk Access: Find file -> Slice 1 row -> Collect
            path, local_idx = self._get_file_info(idx)
            
            # scan_parquet + slice is the efficient way to read 1 row without loading whole file
            df = pl.scan_parquet(path).slice(local_idx, 1).collect()
            
            # Polars stores lists as Series object in the dataframe
            audio_list = df['audio'][0].to_list()
            text_list = df['text'][0].to_list()

        # B. Convert to Tensor
        # audio_list is flattened int32 list from the parquet file
        audio_tensor = torch.tensor(audio_list, dtype=torch.long)
        text_tensor = torch.tensor(text_list, dtype=torch.long)

        # C. Reshape Audio: [Flattened] -> [n_codebooks, seq_len]
        # The tokenizer flattened it; we reshape it back.
        audio_tensor = audio_tensor.view(AUDIO_NUM_CODEBOOKS, -1)

        return {"audio": audio_tensor, "text": text_tensor}



def collate_fn(batch: List[dict]):
    """
    Collate function for tokenized audio and text.
    Merges variable-length audio/text into a single padded tensor.
    """
    tokens, tokens_mask = [], []
    for item in batch:
        audio_tokens = item["audio"]  # [n_codebooks, audio_seq_len]
        text_tokens = item["text"]    # [text_seq_len]

        # 1. Add EOS frame to audio
        # Shape [eos] = [n_codebooks, 1]
        eos_frame = torch.zeros(audio_tokens.size(0), 1, dtype=torch.long)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        # 2. Prepare Audio Frame [seq_len, n_codebooks + 1]
        # We add +1 dim for the text token slot (which is empty/0 for audio frames)
        T_audio = audio_tokens.size(1)
        audio_frame = torch.zeros(T_audio, AUDIO_NUM_CODEBOOKS + 1 , dtype=torch.long)

        # Fill audio columns
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)

        # Mask: Valid audio frames
        audio_frame_mask = torch.zeros(T_audio, AUDIO_NUM_CODEBOOKS + 1, dtype=torch.bool)
        audio_frame_mask[:, :-1] = True

        # 3. Prepare Text Frame [seq_len, n_codebooks + 1]
        # Text goes into the last column (-1)
        T_text = len(text_tokens)
        text_frame = torch.zeros(T_text, AUDIO_NUM_CODEBOOKS + 1, dtype=torch.long)
        text_frame[:, -1] = text_tokens

        # Mask: Valid text frames
        text_frame_mask = torch.zeros(T_text, AUDIO_NUM_CODEBOOKS + 1, dtype=torch.bool)
        text_frame_mask[:, -1] = True

        # 4. Concatenate: Text first, then audio
        tokens.append(torch.cat([text_frame, audio_frame], dim=0))
        tokens_mask.append(torch.cat([text_frame_mask, audio_frame_mask], dim=0))


    # 5. Pad sequences to max length in batch
    tokens = pad_sequence(tokens, batch_first=True) # default padding val is 0
    tokens_mask = pad_sequence(tokens_mask, batch_first=True, padding_value=False)

    return tokens, tokens_mask


# Example
# batch_size = 4
# → bins:
# bin 1: [10,11,12,13]
# bin 2: [200,201,202,203]

# Each bin = one batch returned by the sampler

class BucketSampler(Sampler):
    """
    Groups samples of similar lengths into bins to minimize padding.
    """
    def __init__(
            self, lengths: List[int], batch_size: int, shuffle: bool = True,
            is_infinite: bool = True, random_seed: int = 42
        ):
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.is_infinite = is_infinite
            self.random_seed = random_seed
            self.local_step = 0
            self.bins = self._create_bins(lengths, batch_size)

    def _create_bins(self, lengths: List[int], batch_size: int) -> List[List[int]]:
        # Sort indices by length to group similar sized samples
        indices_with_lengths = sorted(enumerate(lengths), key=lambda x: x[1]) # indices_with_lengths =  [ (1, 3),   # shortest  (2, 7),  (0, 10)   # longest  ] 
        bins, current_bin = [], []

        for idx, _ in indices_with_lengths:
            current_bin.append(idx)
            if len(current_bin) >= batch_size:
                bins.append(current_bin)
                current_bin = []
        
        if current_bin: # taking care of the last bin if it has less than batch_size samples
            bins.append(current_bin)
        
        return bins


    def _shuffle_bins(self, epoch: int):
        # Optional: shuffle within bins so batches aren't identical every epoch
        rng = np.random.RandomState(epoch + self.random_seed)
        rng.shuffle(self.bins)
        for bin_ in self.bins:
            rng.shuffle(bin_)

    def __iter__(self):
        epoch = 0
        while True:
            if self.shuffle:
                self._shuffle_bins(epoch)

            for bin_indices in self.bins:
                yield bin_indices
                self.local_step += 1
            
            if not self.is_infinite:
                break
            epoch += 1

    def __len__(self):
        return len(self.bins)


def create_dataloaders(
    token_dataset_dir: str,
    batch_size: int,
    infinite_train: bool = False,
    load_in_memory: bool = False,
    num_workers: int = 4,
):
    """
    Creates training and validation dataloaders from Parquet directory.
    """
    print(f"Creating dataloaders from {token_dataset_dir}...")
    
    # 1. Train Dataset
    trainset = ParquetTokenizedDataset(token_dataset_dir, split="train", load_in_memory=load_in_memory)
    
    # 2. Validation Dataset (Optional)
    valset = ParquetTokenizedDataset(token_dataset_dir, split="validation", load_in_memory=load_in_memory)
    
    # Warning if validation is empty (e.g. user only has 'test')
    if valset.total_len == 0:
        # Try 'test' split if validation is empty
        print("Validation split empty, checking for 'test' split...")
        valset = ParquetTokenizedDataset(token_dataset_dir, split="test", load_in_memory=load_in_memory)

    # 3. Train Loader
    if trainset.total_len > 0:
        trainsampler = BucketSampler(
            lengths=trainset._lengths, batch_size=batch_size,
            is_infinite=infinite_train, shuffle=True
        )
        trainloader = DataLoader(
            trainset, batch_sampler=trainsampler,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
    else:
        print("WARNING: Train set is empty!")
        trainloader = None

    # 4. Val Loader
    valloader = None
    if valset.total_len > 0:
        valsampler = BucketSampler(
            lengths=valset._lengths, batch_size=batch_size,
            is_infinite=False, shuffle=False
        )
        valloader = DataLoader(
            valset, batch_sampler=valsampler,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
    else:
        print("Validation/Test set empty. valloader will be None.")

    return trainloader, valloader