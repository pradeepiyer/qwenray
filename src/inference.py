#!/usr/bin/env python3
"""Modal inference endpoint for the fine-tuned Qwen Ray model."""

import modal

app = modal.App("qwenray-inference")
volume = modal.Volume.from_name("qwenray-outputs", create_if_missing=False)

MODEL_PATH = "/outputs/qwen-ray-lora"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# CUDA image with vLLM and PEFT
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "vllm==0.6.6.post1",
        "peft>=0.10.0",
    )
)


@app.cls(
    image=vllm_image,
    gpu="T4",
    timeout=600,
    volumes={"/outputs": volume},
)
class RayExpert:
    """Qwen model fine-tuned on Ray documentation."""

    @modal.enter()
    def load_model(self):
        """Load the model with LoRA adapter."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"Loading LoRA adapter from: {MODEL_PATH}")
        self.model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        self.model.eval()
        print("Model loaded successfully!")

    @modal.method()
    def generate(self, question: str, max_tokens: int = 512) -> str:
        """Generate a response to a Ray-related question."""
        import torch

        # Format as ChatML
        messages = [
            {"role": "system", "content": "You are an expert on Ray distributed computing."},
            {"role": "user", "content": question},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        return response.strip()


@app.local_entrypoint()
def main(question: str = "How do I create a Ray actor?"):
    """Test the model with a question."""
    print(f"Question: {question}\n")
    expert = RayExpert()
    response = expert.generate.remote(question)
    print(f"Response:\n{response}")
