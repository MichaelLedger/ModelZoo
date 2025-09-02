# ModelZoo

This repository contains a collection of pre-trained models and utilities for downloading them.

## Required Models

1. [CLIP-ViT-L-14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) - Vision-Language model
2. [Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) - Large Language Model (~13.5GB)
3. [All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Sentence Transformer (Required for confidence estimation)

## Directory Structure

```
|-- ModelZoo
    |-- CLIP
        |-- clip
            |-- ViT-L-14.pt
    |-- LLM
        |-- vicuna
            |-- vicuna-7b-v1.5
                |-- config.json
                |-- generation_config.json
                |-- pytorch_model-00001-of-00002.bin  (9.98GB)
                |-- pytorch_model-00002-of-00002.bin  (3.50GB)
                |-- pytorch_model.bin.index.json
                |-- tokenizer.model
                |-- tokenizer_config.json
                |-- special_tokens_map.json
    |-- SentenceTransformers
        |-- all-MiniLM-L6-v2
            |-- model.safetensors
            |-- pytorch_model.bin
            |-- config files and optimized versions...
```

## Download Instructions

### Prerequisites

1. Python 3.x
2. Virtual Environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install huggingface_hub
```

### Automated Download

Use the provided download script:

```bash
# Download Vicuna model
python download_model.py --model vicuna --verbose

# Download MiniLM model
python download_model.py --model minilm --verbose

# Download both models
python download_model.py --verbose
```

### Important Notes

1. **Model Sizes**:
   - Vicuna-7B: ~13.5GB total
     - Shard 1: 9.98GB
     - Shard 2: 3.50GB
   - CLIP: ~1GB
   - MiniLM: ~90MB

2. **Download Recovery**:
   If the Vicuna model download fails midway (especially for the second shard), you can use:
   ```bash
   hf download lmsys/vicuna-7b-v1.5 pytorch_model-00002-of-00002.bin --local-dir LLM/vicuna/vicuna-7b-v1.5
   ```

3. **Verification**:
   - Ensure all files are downloaded completely
   - Check file sizes match the expected sizes
   - Verify the directory structure matches the one shown above

4. **Common Issues**:
   - Connection timeouts during large file downloads
   - Insufficient disk space (need at least 15GB free)
   - Missing dependencies
   - Incorrect directory permissions

## Support

If you encounter any issues:
1. Check your internet connection stability
2. Ensure sufficient disk space
3. Verify Python and pip are properly installed
4. Make sure you're using the virtual environment