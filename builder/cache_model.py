import os
import sys
from huggingface_hub import login, snapshot_download

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("ERROR: HF_TOKEN environment variable is missing!", file=sys.stderr)
    sys.exit(1)

print("Authenticating with Hugging Face...")
login(token=hf_token)

print("Downloading FLUX.1-dev snapshot to local cache...")
try:
    # Download weights directly into Hugging Face cache without loading into GPU memory
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.bin"]  # Download safetensors only
    )
    print("FLUX.1-dev weights cached successfully!")
except Exception as e:
    print(f"ERROR downloading model snapshot: {e}", file=sys.stderr)
    sys.exit(1)