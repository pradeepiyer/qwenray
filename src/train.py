#!/usr/bin/env python3
"""Modal training script for fine-tuning Qwen on Ray documentation."""

import modal

# Modal app and volume
app = modal.App("qwenray-train")
volume = modal.Volume.from_name("qwenray-outputs", create_if_missing=True)

# Docker image with Axolotl and local files
axolotl_image = (
    modal.Image.from_registry("winglian/axolotl:main-latest", add_python="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("config/qwen-ray.yaml", remote_path="/config/qwen-ray.yaml")
    .add_local_file("data/ray_dataset_small.jsonl", remote_path="/data/ray_dataset_small.jsonl")
)

VOLUME_PATH = "/outputs"
CONFIG_PATH = "/config/qwen-ray.yaml"
DATA_PATH = "/data/ray_dataset_small.jsonl"


@app.function(
    image=axolotl_image,
    gpu="T4",
    timeout=3600,  # 1 hour
    volumes={VOLUME_PATH: volume},
)
def train():
    """Run Axolotl training."""
    import subprocess

    import yaml

    # Load and modify config to use mounted paths
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Update paths for Modal environment
    config["datasets"][0]["path"] = DATA_PATH
    config["output_dir"] = f"{VOLUME_PATH}/qwen-ray-lora"

    # Write modified config
    modified_config_path = "/tmp/config.yaml"
    with open(modified_config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    print("Starting Axolotl training...")
    subprocess.run(
        ["accelerate", "launch", "-m", "axolotl.cli.train", modified_config_path],
        check=True,
    )

    # Commit volume to persist outputs
    volume.commit()

    print(f"Training complete! Model saved to {VOLUME_PATH}/qwen-ray-lora")
    return {"status": "success", "output_dir": f"{VOLUME_PATH}/qwen-ray-lora"}


@app.function(
    image=axolotl_image,
    volumes={VOLUME_PATH: volume},
)
def list_outputs():
    """List files in the output volume."""
    import os

    outputs = []
    for root, _dirs, files in os.walk(VOLUME_PATH):
        for file in files:
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            outputs.append({"path": path, "size": size})

    return outputs


@app.local_entrypoint()
def main(list_only: bool = False):
    """Run training or list outputs."""
    if list_only:
        outputs = list_outputs.remote()
        print("Output files:")
        for item in outputs:
            print(f"  {item['path']} ({item['size']} bytes)")
    else:
        print("Starting training on Modal...")
        result = train.remote()
        print(f"Result: {result}")
