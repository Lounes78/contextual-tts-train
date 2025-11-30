import os 
import sys 
from dotenv import load_dotenv
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from typing import Union
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

CSM_REPO_PATH = os.getenv("CSM_REPO_PATH")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

sys.path.append(CSM_REPO_PATH)
from generator import Generator, load_llama3_tokenizer
from models import Model, ModelArgs, _create_causal_mask

MIMI_SAMPLE_RATE = int(os.getenv("MIMI_SAMPLE_RATE", 24_000))
BACKBONE_FLAVOR = os.getenv("BACKBONE_FLAVOR", "llama-1B")
DECODER_FLAVOR = os.getenv("DECODER_FLAVOR", "llama-100M")
TEXT_VOCAB_SIZE = int(os.getenv("TEXT_VOCAB_SIZE", 128256))
AUDIO_VOCAB_SIZE = int(os.getenv("AUDIO_VOCAB_SIZE", 2051))
AUDIO_NUM_CODEBOOKS = int(os.getenv("AUDIO_NUM_CODEBOOKS", 32)) 

class WarmupDecayLR(LambdaLR):
    """
    Learning rate scheduler with a linear warmup and specificable decay.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, decay_type: str = "linear"):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_type = decay_type
        super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step: int) -> float:
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                if self.decay_type == "linear":
                    return (self.total_steps - step) / (self.total_steps - self.warmup_steps)
                elif self.decay_type == "constant":
                    return 1.0
                elif self.decay_type == "exponential":
                    return 0.1 ** ((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                elif self.decay_type == "cosine":
                    return 0.5 * (1 + torch.cos(torch.pi * torch.tensor((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))))
                else:
                    raise ValueError(f"Invalid decay type: {self.decay_type}")



def load_tokenizers(device: Union[str, torch.device]):
    """Load text and audio tokenizers."""
    text_tokenizer = load_llama3_tokenizer()

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(AUDIO_NUM_CODEBOOKS)
    audio_tokenizer = mimi

    return text_tokenizer, audio_tokenizer


# designed to be monkey-patched into the Model class
def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
    """
    Handles input shapes [B, Seq, N_Codebooks+1] from dataloader. (Seq = audio_seq_len + text_seq_len) )
    """
    dtype = next(self.parameters()).dtype
    bsz, seq_len, _ = tokens.size()
    device = tokens.device

    # embed tokens
    embeds = self._embed_tokens(tokens) # -> [B, Seq, N_Codebooks+1, D]

    # get targets and codebook embeddings corresponding to audio tokens 
    # Dataloader puts audio in the first column of the mask logic
    audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len] 

    # Audio codebooks are in columns 0:-1, Text is at -1
    target_tokens = tokens[audio_mask][:, :-1]  # [audio_len, n_codebooks]
    c_embeds = embeds[:, :, :-1, :][audio_mask]  # [audio_len, n_codebooks, embed_dim]

    # but ofc retain just non-padding embeddings
    masked_embeds = embeds * tokens_mask.unsqueeze(-1) # [B, Seq, N_Codebooks+1, D]
    h = masked_embeds.sum(dim=2)  # [B, Seq, D]

    # backbone forward pass
    padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]  # [bsz, seq_len]
    padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)  # [bsz, seq_len, seq_len]
    
    backbone_attn_mask = _create_causal_mask(seq_len, device)  # [seq_len, seq_len]
    backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d # [bsz, seq_len, seq_len]
    backbone_attn_mask = backbone_attn_mask | torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(bsz, -1, -1) # to force self-attention to be allowed. | also expand is a virtual operation

    # positional indices for positional embeddings / rotary embeddings
    input_pos = torch.arange(0, seq_len).unsqueeze(0).expand(bsz, seq_len).long().to(device) 

    h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(dtype=dtype) # [B, Seq, D]

    # get backbone embeddings used for audio codebook prediction
    # just moves the mask so you select the backbone states that correspond to predicting audio tokens, not the tokens themselves.
    audio_mask = torch.roll(audio_mask, -1, 1)  # shift audio mask to the right by 1 -> [bsz, seq_len] 
    audio_h = h[audio_mask]  # [audio_len, embed_dim]

    # predict first codebook and compute loss
    c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
    c0_target = target_tokens[:, 0]  # [audio_len]
    c0_loss = F.cross_entropy(c0_logits, c0_target)

    # "compute amortization" (train decoder on random 1/16 subset of audio tokens)
    indices = torch.randperm(c_embeds.size(0))[:c_embeds.size(0) // 16] # shuffle audio token indices then take the first 16 audio tokens

    # Check if we have enough tokens to subsample, otherwise skip or use all
    if indices.numel() > 0:
        c_embeds_sub = c_embeds[indices][:, :-1, :] # [audio_len//16, n_codebooks-1, embed_dim]
        audio_h_sub = audio_h[indices]  # [audio_len//16, embed_dim]
        target_tokens_sub = target_tokens[indices][:, 1:] # [audio_len//16, n_codebooks-1]

        # concatenate backbone embeddings and codebook embeddings for decoder input
        # decoder_embeds) looks like this sequence: [Context, C_0, C_1, \dots, C_{30}] --> So the context feeds the backbone AND the decoder
        decoder_embeds = torch.cat([audio_h_sub.unsqueeze(1), c_embeds_sub], dim=1)  # [audio_len//16, n_codebooks, embed_dim] -> audio_len//16 is like batch size for decoder, n_codebooks is seq len for decoder
        N, n_codebooks, _ = decoder_embeds.size() # ofc N = audo_len//16

        # position embeddings for decoder
        c_pos = torch.arange(0, n_codebooks).unsqueeze(0).expand(N, n_codebooks).long().to(device)

        decoder_causal_mask = _create_causal_mask(decoder_embeds.size(1), device).expand(N, -1, -1)  # [N, n_codebooks, n_codebooks]

        decoder_h = self.decoder(self.projection(decoder_embeds), input_pos=c_pos, mask=decoder_causal_mask).to(dtype=dtype)  # [N, n_codebooks, D]
        # torch.einsum(...) It effectively runs 31 different linear classifiers in parallel, one for each codebook index.
        c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)  # [N, n_codebooks-1, audio_vocab_size]

        c_loss = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), target_tokens_sub.reshape(-1))

    else:
        # Fallback if sequence is too short for subsampling
        c_loss = torch.tensor(0.0, device=device, dtype=dtype)

    loss = 2 * ( (1 - self.decoder_loss_weight) * c0_loss + self.decoder_loss_weight * c_loss) 
    return loss



def init_weights(model: nn.Module):
    """
    Initialize the weights of the model.
    """
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.data)
    
    model.apply(_init_weights)
    nn.init.xavier_uniform_(model.audio_head) # Just to be sure :/
    return model



def load_model(model_name_or_checkpoint_path: Union[str, Path] = None, device: Union[str, torch.device] = 'cuda', decoder_loss_weight: float = 0.5) -> Model:
    """
    Load model, add forward method, and move to device.
    """
    if model_name_or_checkpoint_path is None or str(model_name_or_checkpoint_path).endswith(".pt"):
        # Training from scratch or using local checkpoint
        config = ModelArgs(
            backbone_flavor=BACKBONE_FLAVOR,
            decoder_flavor=DECODER_FLAVOR,
            text_vocab_size=TEXT_VOCAB_SIZE,
            audio_vocab_size=AUDIO_VOCAB_SIZE,
            audio_num_codebooks=AUDIO_NUM_CODEBOOKS
        )
        model = Model(config)

        if model_name_or_checkpoint_path:
            print(f"Loading checkpoint from {model_name_or_checkpoint_path}")
            state_dict = torch.load(model_name_or_checkpoint_path, map_location='cpu')['Model']
            model.load_state_dict(state_dict)
        else:
            print('Initializing model from scratch')
            model = init_weights(model)
    else:
        # Huggingface model name
        print(f"Loading pretrained model: {model_name_or_checkpoint_path}")
        model = Model.from_pretrained(model_name_or_checkpoint_path)
    
    model.decoder_loss_weight = decoder_loss_weight # This parameter balances the two different objectives the model is trained on simultaneously
    model.forward = types.MethodType(forward, model) # add the forward method to the model
    model = model.to(device=device)

    return model


def reset_caches(model: Model):
    """Reset the caches of the model (used after each generation)."""
    model.reset_caches()
    for module in model.modules():
        if hasattr(module, "cache_enabled"):
            module.cache_enabled = False
        if hasattr(module, "kv_cache"):
            module.kv_cache = None


def custom_generator_init(self, model: Model, audio_tokenizer: torch.nn.Module, text_tokenizer):
    """Custom __init__ for the Generator class (from sesame csm repo)."""
    self._model = model
    self._model.setup_caches(1)
    
    self._text_tokenizer = text_tokenizer
    
    device = next(model.parameters()).device
    self._audio_tokenizer = audio_tokenizer.to(device=device)
    self.sample_rate = MIMI_SAMPLE_RATE
    self.device = device
    
    self._watermarker = None  


def generate_audio(model, audio_tokenizer, text_tokenizer, text, speaker_id, device, use_amp=True, max_audio_length_ms=10_000):
    """" Generate audio from text """
    model.eval()
    # Patch the Generator's __init__ method on the fly 
    Generator.__init__ = types.MethodType(custom_generator_init, Generator)
    generator = Generator(model, audio_tokenizer, text_tokenizer)

    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        audio = generator.generate(
            text = text,
            speaker = speaker_id,
            context = [],
            max_audio_length_ms = max_audio_length_ms,
        )
        audio = audio.squeeze().cpu().numpy()

    reset_caches(model)
    return audio


def validate(model, valloader, device, use_amp=True):
    """ Validate the model on the validation set """
    model.eval()
    val_losses = []
    
    if valloader is None or len(valloader) == 0:
        print("No validation data provided.")
        return float("inf")



    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        for val_tokens, val_tokens_mask in valloader:
            val_tokens = val_tokens.to(device)
            val_tokens_mask = val_tokens_mask.to(device)
            val_loss = model(val_tokens, val_tokens_mask).item()
            val_losses.append(val_loss)

    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss



# """
# Simplified utilities for CSM fine-tuning.
# """
# import os
# import sys
# import types
# import torch
# import torch.nn.functional as F
# from pathlib import Path
# from typing import Union
# from torch.optim.lr_scheduler import LambdaLR
# from torch import nn

# from huggingface_hub import hf_hub_download
# from moshi.models import loaders

# # Handle both direct execution and module import
# try:
#     from . import config
# except ImportError:
#     # Direct execution fallback
#     import config

# # Add CSM repo to path
# CSM_PATH = os.path.expanduser(config.CSM_REPO_PATH)
# sys.path.append(CSM_PATH)

# from generator import Generator, load_llama3_tokenizer
# from models import Model, ModelArgs, _create_causal_mask


# class WarmupDecayLR(LambdaLR):
#     """Learning rate scheduler with linear warmup and decay."""
    
#     def __init__(self, optimizer, warmup_steps: int, total_steps: int, decay_type: str = "linear"):
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.decay_type = decay_type
#         super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

#     def lr_lambda(self, step: int) -> float:
#         if step < self.warmup_steps:
#             return step / self.warmup_steps
#         else:
#             if self.decay_type == "linear":
#                 return (self.total_steps - step) / (self.total_steps - self.warmup_steps)
#             elif self.decay_type == "constant":
#                 return 1.0
#             elif self.decay_type == "exponential":
#                 return 0.1 ** ((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
#             elif self.decay_type == "cosine":
#                 return 0.5 * (1 + torch.cos(torch.pi * torch.tensor((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))))
#             else:
#                 raise ValueError(f"Invalid decay type: {self.decay_type}")


# def load_tokenizers(device: Union[str, torch.device]):
#     """Load text and audio tokenizers."""
#     text_tokenizer = load_llama3_tokenizer()
#     mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
#     mimi = loaders.get_mimi(mimi_weight, device=device)
#     mimi.set_num_codebooks(config.AUDIO_NUM_CODEBOOKS)
#     audio_tokenizer = mimi
#     return text_tokenizer, audio_tokenizer


# def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
#     """
#     Forward pass for Sesame's CSM model.
#     This will be added to the model with `model.forward = types.MethodType(forward, model)`

#     Args:
#         tokens: (batch_size, seq_len, n_codebooks+1)
#         tokens_mask: (batch_size, seq_len, n_codebooks+1)
#     """
#     dtype = next(self.parameters()).dtype
#     bsz, seq_len, _ = tokens.size()
#     device = tokens.device

#     # embed tokens
#     embeds = self._embed_tokens(tokens)

#     # get targets and codebook embeddings corresponding to audio tokens
#     audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len]
#     target_tokens = tokens[audio_mask][:, :-1]  # [audio_len, n_codebooks]
#     c_embeds = embeds[:, :, :-1, :][audio_mask]  # [audio_len, n_codebooks, embed_dim] 

#     # retain just non-padding embeddings
#     masked_embeds = embeds * tokens_mask.unsqueeze(-1)
#     h = masked_embeds.sum(dim=2)

#     # backbone forward pass
#     padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]  # [bsz, seq_len]
#     backbone_attn_mask = _create_causal_mask(seq_len, device)  # [seq_len, seq_len]
#     padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)  # [bsz, seq_len, seq_len]
#     backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
#     backbone_attn_mask = backbone_attn_mask | torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(bsz, -1, -1)
#     input_pos = torch.arange(0, seq_len).unsqueeze(0).expand(bsz, seq_len).long().to(device)
#     h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(dtype=dtype)

#     # get backbone embeddings used for audio codebook prediction
#     audio_mask = torch.roll(audio_mask, -1, 1)  # shift audio mask to the right by 1
#     audio_h = h[audio_mask]  # [audio_len, embed_dim]

#     # predict first codebook and compute loss
#     c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
#     c0_target = target_tokens[:, 0]  # [audio_len]
#     c0_loss = F.cross_entropy(c0_logits, c0_target)

#     # "compute amortization" (train decoder on random 1/16 subset of audio tokens)
#     indices = torch.randperm(c_embeds.size(0))[: c_embeds.size(0) // 16]
#     c_embeds = c_embeds[indices][:, :-1, :]  # [audio_len//16, n_codebooks-1, embed_dim]
#     audio_h = audio_h[indices]  # [audio_len//16, embed_dim]
#     target_tokens = target_tokens[indices][:, 1:]  # [audio_len//16, n_codebooks-1]

#     # concatenate backbone embeddings and codebook embeddings for decoder input
#     decoder_embeds = torch.cat(
#         [audio_h.unsqueeze(1), c_embeds], dim=1
#     )  # [audio_len//16, n_codebooks, embed_dim]
#     N, n_codebooks, _ = decoder_embeds.size()
#     c_pos = torch.arange(0, n_codebooks).unsqueeze(0).expand(N, n_codebooks).long().to(device)

#     decoder_causal_mask = _create_causal_mask(decoder_embeds.size(1), device).expand(N, -1, -1)
#     decoder_h = self.decoder(self.projection(decoder_embeds), input_pos=c_pos, mask=decoder_causal_mask).to(dtype=dtype)
#     c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)

#     c_loss = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), target_tokens.reshape(-1))

#     loss = 2 * ((1 - self.decoder_loss_weight) * c0_loss + self.decoder_loss_weight * c_loss)
#     return loss


# def init_weights(model: nn.Module):
#     """Initialize the weights of the model."""
#     def _init_weights(m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Embedding):
#             nn.init.normal_(m.weight, mean=0.0, std=0.02)
#         elif isinstance(m, nn.Parameter):
#             nn.init.xavier_uniform_(m.data)

#     model.apply(_init_weights)
#     # Special handling for audio_head because it's nn.Parameter directly
#     nn.init.xavier_uniform_(model.audio_head)
#     return model


# def load_model(device: Union[str, torch.device] = 'cuda') -> Model:
#     """Load model, add forward method, and move to device."""
    
#     if config.TRAIN_FROM_SCRATCH:
#         # Training from scratch
#         model_config = ModelArgs(
#             backbone_flavor=config.BACKBONE_FLAVOR,
#             decoder_flavor=config.DECODER_FLAVOR,
#             text_vocab_size=config.TEXT_VOCAB_SIZE,
#             audio_vocab_size=config.AUDIO_VOCAB_SIZE,
#             audio_num_codebooks=config.AUDIO_NUM_CODEBOOKS
#         )
#         model = Model(model_config)
#         model = init_weights(model)
#     else:
#         # Load pretrained model
#         model = Model.from_pretrained(config.MODEL_NAME)

#     model.decoder_loss_weight = config.DECODER_LOSS_WEIGHT
#     model.forward = types.MethodType(forward, model)  # add the forward method to the model
#     model = model.to(device=device)
#     return model


# def reset_caches(model: Model):
#     """Reset the caches of the model (used after each generation)."""
#     model.reset_caches()
#     for module in model.modules():
#         if hasattr(module, "cache_enabled"):
#             module.cache_enabled = False
#         if hasattr(module, "kv_cache"):
#             module.kv_cache = None


# def custom_generator_init(self, model: Model, audio_tokenizer: torch.nn.Module, text_tokenizer):
#     """Custom __init__ for the Generator class (from sesame csm repo)."""
#     self._model = model
#     self._model.setup_caches(1)
    
#     self._text_tokenizer = text_tokenizer
    
#     device = next(model.parameters()).device
#     self._audio_tokenizer = audio_tokenizer.to(device=device)
#     self.sample_rate = config.MIMI_SAMPLE_RATE
#     self.device = device
    
#     self._watermarker = None  # No watermarking


# def generate_audio(model, audio_tokenizer, text_tokenizer, text, speaker_id, device,
#                   use_amp=True, max_audio_length_ms=10_000):
#     """Generate audio from text."""
#     model.eval()
#     Generator.__init__ = types.MethodType(custom_generator_init, Generator)
#     generator = Generator(model, audio_tokenizer, text_tokenizer)
    
#     with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
#         audio = generator.generate(
#             text=text,
#             speaker=speaker_id,
#             context=[],
#             max_audio_length_ms=max_audio_length_ms,
#         )
#         audio = audio.squeeze().cpu().numpy()
    
#     reset_caches(model)
#     return audio


# def validate(model, valloader, device, use_amp=True):
#     """Validate the model on the validation set."""
#     model.eval()
#     val_losses = []
#     with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
#         for val_tokens, val_tokens_mask in valloader:
#             val_tokens = val_tokens.to(device)
#             val_tokens_mask = val_tokens_mask.to(device)
#             val_loss = model(val_tokens, val_tokens_mask).item()
#             val_losses.append(val_loss)
    
#     avg_val_loss = sum(val_losses) / len(val_losses)
#     return avg_val_loss