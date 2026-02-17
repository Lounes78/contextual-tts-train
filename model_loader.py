import torch
try:
    from .decoding import AudioDecoder
    from .token_generator import TokenGenerator
except ImportError:
    from decoding import AudioDecoder
    from token_generator import TokenGenerator
import sys
import os

# Add parent directory to path to import models if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models import Model, ModelArgs
except ImportError:
    pass

def load_csm_1b(device: str = "cuda") -> TokenGenerator:
    """
    Load the CSM-1B model with optimizations and return a TokenGenerator instance.
    """
    # Enable all CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    print("Loading CSM-1B model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    model = Model.from_pretrained("sesame/csm-1b")
    dtype = torch.float16
    model.to(device=device, dtype=dtype)
    
    print("Model loaded. Configuring compilation...")
    
    # Configure inductor to cache compiled kernels and graphs to disk
    import torch._inductor.config as inductor_config
    inductor_config.fx_graph_cache = True
    inductor_config.triton.unique_kernel_names = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.max_autotune = True
    inductor_config.epilogue_fusion = True
    inductor_config.triton.cudagraphs = False  # Disabled due to GH200 unified memory issues

    # Compile the CSM backbone and decoder with torch.compile for maximum throughput.
    # Requires torch>=2.7.0 with proper aarch64 triton support.
    # First run will be slow (compilation), subsequent runs use cached kernels.
    try:
        print("Compiling backbone with torch.compile (first run will be slow)...")
        model.backbone = torch.compile(
            model.backbone,
            mode='max-autotune',
            fullgraph=False,
            backend='inductor',
            dynamic=True,  # Handle dynamic sequence lengths
        )
        print("Compiling decoder...")
        model.decoder = torch.compile(
            model.decoder,
            mode='max-autotune',
            fullgraph=False,
            backend='inductor',
            dynamic=True,
        )
        print("Compilation configured (will trigger on first inference).")
    except Exception as e:
        print(f"torch.compile failed, falling back to eager mode: {e}")

    print("Creating AudioDecoder...")
    # Get num_codebooks from model config if available, otherwise default to 8
    num_codebooks = getattr(model.config, 'audio_num_codebooks', 8)
    audio_decoder = AudioDecoder(device=device, num_codebooks=num_codebooks)
    
    print("Creating TokenGenerator...")
    generator = TokenGenerator(model, audio_decoder)
    
    # Optional warmup could go here
    
    return generator
