from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class Segment:
    speaker: int
    text: str
    sample_rate = 24_000
    audio: torch.Tensor

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 0, value: float = 0.0) -> torch.Tensor:
    """
    Pad a tensor so that its size in `dim` is a multiple of `multiple`.
    """
    size = x.size(dim)
    rem = size % multiple
    if rem == 0:
        return x
    
    pad_size = multiple - rem
    
    # Construct padding tuple for F.pad (last_dim_left, last_dim_right, 2nd_last_left, ...)
    pad_args = [0] * (x.ndim * 2) 
    
    # Calculate index for right padding on dimension `dim`
    # F.pad expects list reversed from dimensions: (last dim, 2nd last dim, ...)
    # If dim=0 on (A, B), user wants to pad A. A is 2nd last (index 1 from end).
    # pad args for last dim are at 0,1. 2nd last dim at 2,3.
    # Formula: index = (ndim - 1 - dim) * 2 + 1  (for right padding)
    
    idx = (x.ndim - 1 - dim) * 2 + 1
    pad_args[idx] = pad_size
    
    return F.pad(x, tuple(pad_args), value=value)
