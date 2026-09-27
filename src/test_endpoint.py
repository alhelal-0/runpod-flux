import time
import requests
import base64

RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY"
ENDPOINT_ID = "YOUR_ENDPOINT_ID"

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "prompt": "A cinematic shot of a futuristic city at sunset, highly detailed, 8k resolution",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": 42
    }
}

response = requests.post(url, headers=headers, json=payload)
job_data = response.json()
print("Job Submitted:", job_data)

job_id = job_data.get("id")
status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"

while True:
    res = requests.get(status_url, headers=headers).json()
    status = res.get("status")
    print(f"Status: {status}")
    
    if status == "COMPLETED":
        image_b64 = res["output"]["image"].split(",")[1]
        with open("output_flux.png", "wb") as f:
            f.write(base64.b64decode(image_b64))
        print("Success! Saved generated image to output_flux.png")
        break
    elif status in ["FAILED", "CANCELLED"]:
        print("Job Failed:", res)
        break
        
    time.sleep(5)


    #python test_endpoint.py