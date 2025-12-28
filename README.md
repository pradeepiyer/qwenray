# qwenray

Fine-tune Qwen 2.5-0.5B as a Ray distributed computing expert using Modal.

## Requirements

- Modal account
- OpenAI API key (`OPENAI_API_KEY` env var)
- Ray repo cloned at `~/ray`
- uv package manager

## Setup

```bash
uv sync
modal token new  # Authenticate with Modal
```

## Usage

### 1. Generate Training Data

```bash
uv run python src/data_prep.py --samples 500
```

Uses OpenAI to generate Q&A pairs from Ray documentation.

### 2. Train on Modal

```bash
uv run modal run src/train.py
```

- Runs on Modal T4 GPU
- Uses Axolotl with LoRA
- Model saved to Modal volume: `qwenray-outputs`

### 3. Run Inference

```bash
uv run modal run src/inference.py --question "How do I create a Ray actor?"
```

- Loads fine-tuned model from Modal volume
- Uses PEFT to merge LoRA adapter
- Runs on Modal A10G with vLLM

## Project Structure

```
src/data_prep.py     - Generate Q&A from Ray docs via OpenAI
src/train.py         - Modal training with Axolotl + LoRA
src/inference.py     - Modal inference endpoint
config/qwen-ray.yaml - Axolotl training config
```
