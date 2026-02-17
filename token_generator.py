import torch
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from typing_extensions import OrderedDict
from typing import List, Tuple, Optional, Callable
import time
import sys
import os

# Add parent directory to path to import models if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models import Model
except ImportError:
    pass

try:
    from .common import Segment
    from .decoding import AudioDecoder
except ImportError:
    from common import Segment
    from decoding import AudioDecoder

def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer

class TokenGenerator:
    def __init__(self, model: Model, audio_decoder: AudioDecoder):
        self._model = model
        self.audio_decoder = audio_decoder
        self.device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        
        # Setup caches
        self._model.setup_caches(1)
        self._cache = OrderedDict()
        self._text_token_cache = {}
        self._segment_cache = {}  # Cache tokenized segments (text+audio) to avoid re-encoding
        
        self._text_tokenizer = load_llama3_tokenizer()
        self._num_codebooks = audio_decoder._num_codebooks
        
        # Generation parameters
        self._stream_buffer_size = 20
        self.max_seq_len = 2048

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text segment with caching optimization.
        """
        cache_key = f"{speaker}:{text}"
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        result = (text_frame, text_frame_mask)
        self._text_token_cache[cache_key] = result
        return result

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cache by (text, speaker, audio_data_ptr) to avoid re-encoding prompt audio
        cache_key = (segment.text, segment.speaker, segment.audio.data_ptr())
        if cache_key in self._segment_cache:
            return self._segment_cache[cache_key]

        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self.audio_decoder.encode(segment.audio)
        
        # --- SHAPE STABILIZATION STRATEGY ---
        # To minimize recompilations, we can pad context segments to multiples (e.g. 16, 32)
        # However, for pure autoregressive generation, the *history* length matters most.
        # If we concatenate tokens, the total length changes.
        # Compiling with dynamic=True handles this, but minimizing unique shapes still helps.
        
        total_len = text_tokens.size(0) + audio_tokens.size(0)
        
        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len

            if text_tokens.size(0) > overflow:
                text_tokens = text_tokens[overflow:]
                text_masks = text_masks[overflow:]
            else:
                audio_overflow = overflow - text_tokens.size(0)
                text_tokens = text_tokens[0:0] # empty
                text_masks = text_masks[0:0]
                audio_tokens = audio_tokens[audio_overflow:]
                audio_masks = audio_masks[audio_overflow:]

        result = (torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0))
        self._segment_cache[cache_key] = result
        return result

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        """
        Generate audio tokens stream.
        Yields audio chunks (decoded via audio_decoder).
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]
        print(f"[Gen-{session_id}] Starting generate_stream for text: '{text[:30]}...'")
        yield_count = 0
        device = self.device
        is_cuda = device.type == 'cuda'
        
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self._model.reset_caches()

        # Calculate reasonable max length based on text
        # Rough estimate: ~10 chars/second of speech, 80ms per frame
        # So chars * 0.1 / 0.08 = chars * 1.25 frames, with 2x safety margin
        estimated_frames = max(int(len(text) * 2.5), 50)
        max_generation_len = min(int(max_audio_length_ms / 80), estimated_frames)
        
        # Build prompt tokens
        tokens, tokens_mask = [], []
        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(device)

        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=device).unsqueeze(0).long()

        # Pre-allocate constants on device once
        zeros_1_1 = torch.zeros(1, 1, dtype=torch.long, device=device)
        zeros_mask_1_1 = torch.zeros(1, 1, dtype=torch.bool, device=device)

        # Streaming buffer config
        # Smooth progressive buffering to avoid stuttering
        buffer_size = 15  # Small initial buffer for low latency
        frame_buffer = []
        first_chunk_delivered = False
        second_chunk_delivered = False

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample, dtype=torch.bool)
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        with self.audio_decoder.streaming_context(1):
            for i in range(max_generation_len):
                # Generate one frame — the backbone + decoder are the hot path
                sample = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                )
                
                # EOS check: stop immediately if frame is all zeros (like reference implementation)
                if torch.all(sample == 0):
                    # Process any buffered frames before stopping
                    if frame_buffer:
                        yield_count += 1
                        print(f"[Gen-{session_id}] EOS: Flushing {len(frame_buffer)} buffered frames as chunk #{yield_count}")
                        frames_stacked = torch.stack(frame_buffer).squeeze(1)
                        audio_chunk = self.audio_decoder.decode(frames_stacked)
                        cpu_chunk = audio_chunk.cpu() if is_cuda else audio_chunk
                        if on_chunk_generated:
                            on_chunk_generated(cpu_chunk)
                        else:
                            yield cpu_chunk
                        frame_buffer = []  # CRITICAL: Clear buffer to prevent duplicate in final handler
                    break
                
                update_tokens(sample)
                frame_buffer.append(sample)

                # Flush buffer when full
                if len(frame_buffer) >= buffer_size:
                    yield_count += 1
                    print(f"[Gen-{session_id}] Yield #{yield_count}: Flushing {len(frame_buffer)} frames")
                    frames_stacked = torch.stack(frame_buffer).squeeze(1)  # (T, K)
                    audio_chunk = self.audio_decoder.decode(frames_stacked)
                    
                    # Transfer to CPU just before yielding
                    cpu_chunk = audio_chunk.cpu() if is_cuda else audio_chunk
                    print(f"[Gen-{session_id}] About to yield chunk #{yield_count}: {len(cpu_chunk)} samples")
                    
                    frame_buffer = []
                    
                    if on_chunk_generated:
                        on_chunk_generated(cpu_chunk)
                        print(f"[Gen-{session_id}] Called callback for chunk #{yield_count}, continuing loop...")
                    else:
                        print(f"[Gen-{session_id}] YIELDING chunk #{yield_count} now...")
                        yield cpu_chunk
                        print(f"[Gen-{session_id}] RETURNED from yield #{yield_count}, continuing loop...")

                    if not first_chunk_delivered:
                        # After first chunk, increase gradually to avoid stuttering
                        buffer_size = 25  # Moderate increase
                        first_chunk_delivered = True
                    elif not second_chunk_delivered:
                        # After second chunk, use larger buffer for efficiency
                        buffer_size = 40  # Final buffer size
                        second_chunk_delivered = True

            # Flush remaining frames (already checked for EOS above)
            if frame_buffer:
                yield_count += 1
                print(f"[Gen-{session_id}] Final chunk #{yield_count}: {len(frame_buffer)} frames")
                frames_stacked = torch.stack(frame_buffer).squeeze(1)
                audio_chunk = self.audio_decoder.decode(frames_stacked)
                cpu_chunk = audio_chunk.cpu() if is_cuda else audio_chunk
                    
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                    print(f"[Gen-{session_id}] Called callback for final chunk #{yield_count}: {len(cpu_chunk)} samples")
                else:
                    print(f"[Gen-{session_id}] YIELDING final chunk #{yield_count}: {len(cpu_chunk)} samples")
                    yield cpu_chunk
                    print(f"[Gen-{session_id}] RETURNED from final yield #{yield_count}")
            
            print(f"[Gen-{session_id}] Generator complete. Total chunks: {yield_count}")
