# μGATO Project Assistant Guidelines

## Project Overview

μGATO (Mini Unofficial GATO) is an educational implementation of DeepMind's GATO paper. It demonstrates how to unify different data modalities (text, images, actions) into a single sequence model (typically a transformer).

## Key Files and Their Purposes

- `mugato/mugato.py`: Core model implementation combining tokenizer, embedder, and transformer
- `mugato/tokenizer.py`: Handles encoding/decoding of text, discrete values, and continuous values
- `mugato/nano_gpt.py`: Transformer implementation based on nanoGPT
- `mugato/models/transformer.py`: Alternative transformer implementation
- `mugato/data/`: Dataset implementations for various tasks (text, vision, RL)
- `mugato/train.py`: Training script within the package
- `train.py`: Main training script with distributed training support
- `demo.ipynb`: Interactive demonstration notebook
- `mugato.ipynb`: Detailed walkthrough of the implementation
- `token_and_embed.ipynb`: Exploration of tokenization and embedding
- `configurator.py`: Configuration management utilities

## Technical Details

### Data Shape Convention
All data follows the shape: `(Batch, Episodes, Tokens, Channels)`
- Text: Episodes=1, Channels=1
- Images: Tokens are 16x16 patches, Channels=768 (ResNet embeddings)
- Actions: Episodes vary by task, Channels=1

### Tokenization
- Text: tiktoken r50k_base (50,257 tokens)
- Discrete values: Additional 1,024 tokens
- Continuous values: Encoded as special tokens + 8-bit mu-law encoding
- Total vocabulary: 51,281 tokens

### Model Architecture
- Configurable transformer (layers, heads, dimensions)
- ResNetV2 for image embeddings
- Supports multiple datasets simultaneously via `create_combined_dataloader`

## Common Tasks and Questions

When asked about:
- **Model architecture**: Reference `mugato/mugato.py` and explain the flow: tokenize → embed → transformer → decode
- **Adding new datasets**: Point to existing examples in `mugato/data/` and the three required functions
- **Training configurations**: Refer to `train.py` command-line arguments
- **Data format questions**: Emphasize the unified shape and how different modalities are converted
- **Performance**: Remind that this is educational - clarity over performance

## Development Commands

**IMPORTANT**: This project uses `uv` for dependency management. Always use `uv` commands instead of pip/python directly.

If the user needs to run commands (only mention when asked):
- Install: `uv pip install -e .`
- Train: `uv run python train.py --dataset shakespeare` (or other datasets)
- Test: `uv run pytest tests/`
- Type checking: `uv run mypy mugato/`
- Linting: `uv run ruff check mugato/`
- Run any Python script: `uv run python <script.py>`
- Install dependencies: `uv pip install <package>`
- Sync dependencies from pyproject.toml: `uv pip sync`

## Development Best Practices

- Run `uv run ruff check --fix . --exclude '*.ipynb' after every change and make sure your code passes *all* checks.

## Important Notes

1. This is an educational implementation - prioritize clarity over performance
2. The project demonstrates GATO's key insight: unifying diverse data types
3. All modifications should maintain the consistent data shape convention
4. The codebase is designed for experimentation and understanding
