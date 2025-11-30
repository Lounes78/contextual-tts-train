"""
Main script to run the custom CSM fine-tuning pipeline.
Simple workflow: pretokenize -> train
"""

import sys
from pathlib import Path

def main():
    """Main execution function."""
    
    print("=== Custom CSM Fine-tuning Pipeline ===")
    print("1. Pre-tokenization")
    print("2. Training")
    print()
    
    # Import with fallback for direct execution
    try:
        from . import pretokenize as pretokenize_module
        from . import train as train_module
        from . import config
    except ImportError:
        import pretokenize as pretokenize_module
        import train as train_module
        import config
    
    # Check if CSM repo exists
    csm_path = Path(config.CSM_REPO_PATH).expanduser()
    if not csm_path.exists():
        print(f"❌ CSM repository not found at: {csm_path}")
        print("Please clone the CSM repository first:")
        print(f"git clone https://github.com/SesameAILabs/csm.git {csm_path}")
        print("git checkout 836f886515f0dec02c22ed2316cc78904bdc0f36")
        return
    
    # Check if Parquet data exists
    parquet_data_path = Path(config.PARQUET_DATA_PATH)
    
    if not parquet_data_path.exists():
        print(f"❌ Parquet data directory not found: {parquet_data_path}")
        print("Please ensure you have the peoples_speech dataset downloaded.")
        return
    
    # Check for train, validation, and test files
    train_files = list(parquet_data_path.glob("train-*.parquet"))
    val_files = list(parquet_data_path.glob("validation-*.parquet"))
    test_files = list(parquet_data_path.glob("test-*.parquet"))
    
    if not train_files or not val_files:
        print(f"❌ Missing required Parquet files in {parquet_data_path}:")
        print(f"   Train files: {len(train_files)} found")
        print(f"   Validation files: {len(val_files)} found")
        print(f"   Test files: {len(test_files)} found")
        print("Expected format: train-*.parquet and validation-*.parquet (test-*.parquet is optional)")
        return
    
    print(f"✅ Found {len(train_files)} train files, {len(val_files)} validation files, and {len(test_files)} test files")
    
    try:
        # Step 1: Pre-tokenization
        # Check for any existing tokenized chunks
        tokenized_dir = Path(config.TOKENIZED_DATA_PATH)
        train_chunks = list(tokenized_dir.glob("train_part_*.parquet"))
        val_chunks = list(tokenized_dir.glob("validation_part_*.parquet"))
        test_chunks = list(tokenized_dir.glob("test_part_*.parquet"))
        
        if not train_chunks or not val_chunks:
            print("🔄 Running pre-tokenization...")
            pretokenize_module.main()
        else:
            print(f"✅ Pre-tokenized data already exists:")
            print(f"   Train chunks: {len(train_chunks)}")
            print(f"   Validation chunks: {len(val_chunks)}")
            print(f"   Test chunks: {len(test_chunks)}")
            print("   Skipping tokenization")
        
        # Step 2: Training
        print("🚀 Starting training...")
        best_val_loss = train_module.train()
        
        print(f"🎉 Training completed successfully!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved to: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()