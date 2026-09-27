import io
import base64
import torch
import runpod
from diffusers import FluxPipeline

# Global variable to hold model across invocations
pipe = None

def init():
    global pipe
    if pipe is None:
        print("Loading FLUX.1-dev into GPU memory...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()  # Saves VRAM when idle
        print("FLUX.1-dev loaded successfully.")

def handler(job):
    """
    Expects job payload format:
    {
        "input": {
            "prompt": "A futuristic city at sunset, highly detailed",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "seed": 42
        }
    }
    """
    job_input = job.get("input", {})
    
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A 'prompt' parameter is required."}

    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    num_inference_steps = job_input.get("num_inference_steps", 28)
    guidance_scale = job_input.get("guidance_scale", 3.5)
    seed = job_input.get("seed", None)

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    try:
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Convert image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image": f"data:image/png;base64,{img_str}",
            "seed": seed
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    init()
    runpod.serverless.start({"handler": handler})