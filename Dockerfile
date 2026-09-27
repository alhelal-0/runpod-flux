FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ src/

CMD ["python", "-u", "src/handler.py"]