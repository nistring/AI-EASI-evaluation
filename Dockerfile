FROM python:3.8-slim
From pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /AI-EASI-evaluation

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip --no-cache-dir install --upgrade pip
RUN pip3 install torchvision --index-url https://download.pytorch.org/whl/cu118
RUN git clone https://github.com/nistring/AI-EASI-evaluation .
COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
