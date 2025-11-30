"""
Direct configuration for CSM fine-tuning - no command line arguments needed.
"""
import os
from pathlib import Path

# Paths
CSM_REPO_PATH = "../csm"  
PARQUET_DATA_PATH = "../data/peoples_speech/dirty"  
TOKENIZED_DATA_PATH = "./tokenized/data/peoples_speech/dirty"
OUTPUT_DIR = "./custom/output"                 

# Model settings
MODEL_NAME = "sesame/csm-1b"  # Pretrained model to finetune
DECODER_LOSS_WEIGHT = 0.5
TRAIN_FROM_SCRATCH = False

# Training hyperparameters
BATCH_SIZE = 8
GRAD_ACC_STEPS = 1
LEARNING_RATE = 3e-5
MAX_GRAD_NORM = 1.3
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.002
LR_DECAY = "linear"  # Options: linear, cosine, constant, exponential

# Training settings
N_EPOCHS = 25
USE_AMP = True  # Automatic Mixed Precision
LOG_EVERY = 10
VAL_EVERY = 100
SAVE_EVERY = 1000
GEN_EVERY = 500

# Generation settings
GEN_SENTENCE = "Bird law in this country is not governed by reason."
GEN_SPEAKER = 999

# Data settings
SAVE_BATCH_SIZE = 1000  # For pretokenization batching
OMIT_SPEAKER_ID = False

# Audio/tokenizer constants
MIMI_SAMPLE_RATE = 24000
AUDIO_NUM_CODEBOOKS = 32
AUDIO_VOCAB_SIZE = 2051
TEXT_VOCAB_SIZE = 128256
BACKBONE_FLAVOR = "llama-1B"
DECODER_FLAVOR = "llama-100M"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TOKENIZED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)