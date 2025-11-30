"""
Custom CSM Fine-tuning Pipeline

A simplified training pipeline for Sesame AI's Conversational Speech Model (CSM).
Based on the sesame-finetune repository but with direct configuration and Parquet storage.
"""

from . import config
from .utils import load_model, load_tokenizers, generate_audio, validate
from .dataloaders import create_dataloaders
from .pretokenize import main as pretokenize
from .train import train

__version__ = "1.0.0"
__all__ = [
    "config", 
    "load_model", 
    "load_tokenizers", 
    "generate_audio", 
    "validate",
    "create_dataloaders", 
    "pretokenize", 
    "train"
]