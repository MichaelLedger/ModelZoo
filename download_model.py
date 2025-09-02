from huggingface_hub import hf_hub_download, snapshot_download
import os
import argparse
import logging
import shutil

# Enable faster downloads with hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def get_dir_size(path):
    """Get the size of a directory in bytes."""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def format_size(size_bytes):
    """Format size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def download_minilm(model_dir, verbose=False):
    """Download the MiniLM model (should be around 90MB)."""
    logging.info("Starting download of all-MiniLM-L6-v2 model...")
    
    try:
        local_dir = snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        size = get_dir_size(local_dir)
        logging.info(f"MiniLM model successfully downloaded to: {os.path.abspath(local_dir)}")
        logging.info(f"Model size: {format_size(size)}")
        return True
    except Exception as e:
        logging.error(f"Error downloading MiniLM model: {str(e)}")
        return False

def download_vicuna(model_dir, verbose=False):
    """Download the Vicuna model (will be several GB)."""
    logging.info("Starting download of Vicuna-7b-v1.5 model files...")
    
    # Create the model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Essential files to download
    files_to_download = [
        "config.json",
        "generation_config.json",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json"
    ]
    
    success = True
    try:
        # Download config and tokenizer files
        for filename in files_to_download:
            logging.info(f"Downloading {filename}...")
            local_file = hf_hub_download(
                repo_id="lmsys/vicuna-7b-v1.5",
                filename=filename,
                local_dir=model_dir
            )
            
        # Download model shards
        for i in range(2):  # Vicuna-7b typically has 2 shards
            shard_file = f"pytorch_model-0000{i+1}-of-00002.bin"
            logging.info(f"Downloading model shard {shard_file} (this may take a while)...")
            local_file = hf_hub_download(
                repo_id="lmsys/vicuna-7b-v1.5",
                filename=shard_file,
                local_dir=model_dir
            )
        
        size = get_dir_size(model_dir)
        logging.info(f"Vicuna model successfully downloaded to: {os.path.abspath(model_dir)}")
        logging.info(f"Model size: {format_size(size)}")
        return True
            
    except Exception as e:
        logging.error(f"Error downloading Vicuna model: {str(e)}")
        return False

def main(verbose=False, model_type=None):
    # Set up logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure base directories exist
    os.makedirs("LLM/vicuna", exist_ok=True)
    os.makedirs("SentenceTransformers", exist_ok=True)
    
    if model_type in [None, 'vicuna']:
        vicuna_success = download_vicuna("LLM/vicuna/vicuna-7b-v1.5", verbose)
    
    if model_type in [None, 'minilm']:
        minilm_success = download_minilm("SentenceTransformers/all-MiniLM-L6-v2", verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--model", choices=['vicuna', 'minilm'], help="Specify which model to download")
    args = parser.parse_args()
    
    main(verbose=args.verbose, model_type=args.model)