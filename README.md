# qwen-anyscale

Fine-tune Qwen 2.5-0.5B as a Ray and Anyscale platform expert using Modal.

## Requirements

- Modal account
- OpenAI API key (`OPENAI_API_KEY` env var)
- Ray repo cloned at `~/ray`
- Anyscale docs cloned at `~/docs`
- uv package manager

## Setup

```bash
uv sync
modal token new  # Authenticate with Modal
```

## Usage

### 1. Generate Training Data

```bash
uv run python src/data_prep.py --sources ray,anyscale --ray-samples 50 --anyscale-samples 50 --output data/test_dataset.jsonl
```

Uses OpenAI to generate Q&A pairs from Ray and Anyscale documentation.

### 2. Train on Modal

```bash
uv run modal run src/train.py
```

- Runs on Modal A10G GPU
- Uses Axolotl with LoRA
- Model saved to Modal volume: `qwen-anyscale-outputs`

### 3. Run Inference

```bash
uv run modal run src/inference.py --question "How do I create a Ray actor?"
```

- Loads fine-tuned model from Modal volume
- Uses PEFT to merge LoRA adapter
- Runs on Modal T4 GPU

## Project Structure

```
src/data_prep.py     - Generate Q&A from Ray + Anyscale docs via OpenAI
src/train.py         - Modal training with Axolotl + LoRA
src/inference.py     - Modal inference endpoint
config/qwen-anyscale.yaml - Axolotl training config
```
