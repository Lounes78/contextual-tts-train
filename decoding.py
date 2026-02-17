import torch
from huggingface_hub import hf_hub_download
# Assuming moshi is installed or available in PYTHONPATH
try:
    from moshi.models import loaders
except ImportError:
    # If not importable directly, maybe assume user has run it from root
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from moshi.models import loaders

from typing import Tuple, List, Optional
import contextlib

class AudioDecoder:
    def __init__(self, device: str, num_codebooks: int = 8):
        self.device = device
        self._num_codebooks = num_codebooks
        
        # Load Mimi model
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(num_codebooks)
        
        self._audio_tokenizer = mimi
        self.sample_rate = mimi.sample_rate

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes audio tensor to tokens.
        Returns (frame_tokens, frame_masks) -- tokens likely (TIME, CODEBOOKS+1)
        """
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device).detach()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
            
        with torch.no_grad():
            audio_tokens = self._audio_tokenizer.encode(audio)[0]
        
        # Limit to the number of codebooks set in MIMI
        audio_tokens = audio_tokens[:self._num_codebooks, :]
        
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        
        # Transpose to (T, K)
        audio_frame = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).bool().to(self.device)
        
        # audio_tokens is (K, T) -> transpose to (T, K) and put in frame
        audio_frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :self._num_codebooks] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    @torch.inference_mode()
    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Decodes frames (tokens) to audio.
        frames: Tensor of shape (T, K) or similar stackable list
        """
        if isinstance(frames, list):
             frames_stacked = torch.stack(frames)
        else:
             frames_stacked = frames
             
        if frames_stacked.numel() == 0:
            return torch.tensor([], device=self.device)

        # Taking only first N codebooks
        frames_reduced = frames_stacked[:, :self._num_codebooks]
        
        # Decode expects (B, K, T)
        # frames_reduced is (T, K) -> unsqueeze -> (1, T, K) -> permute -> (1, K, T)
        mimi_input = frames_reduced.unsqueeze(0).permute(0, 2, 1).contiguous()  # (1, K, T)
        
        # Keep audio on GPU until explicit transfer
        audio = self._audio_tokenizer.decode(mimi_input).squeeze(0).squeeze(0)
        return audio

    def streaming_context(self, batch_size: int = 1):
        return self._audio_tokenizer.streaming(batch_size)
