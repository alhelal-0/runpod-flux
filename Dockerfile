FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Accept Hugging Face Token as build argument for downloading FLUX.1-dev
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Cache FLUX.1-dev model in the image
COPY builder/cache_model.py builder/cache_model.py
RUN python builder/cache_model.py

# Copy application source code
COPY src/ src/

CMD ["python", "-u", "src/handler.py"]
