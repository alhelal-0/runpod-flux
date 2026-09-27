import os
import torch
from diffusers import FluxPipeline

def download_model():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required to download FLUX.1-dev.")
    
    print("Downloading FLUX.1-dev pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    print("Model downloaded and cached successfully.")

if __name__ == "__main__":
    download_model()