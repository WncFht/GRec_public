# -----------------------------------------------------------------------------
# Dockerfile for Unify Multimodal Generative Recommendation
# -----------------------------------------------------------------------------

# --- Base Image ---
# 使用 Ubuntu 22.04 作为基础，它提供了稳定且广泛兼容的环境。
# CUDA 12.4.1 和 cuDNN 为项目提供了必要的 GPU 加速能力。
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# --- Environment Setup ---
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    fish \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 安装 gosu 用于用户切换
RUN apt-get update && apt-get install -y curl gosu && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir packaging==25.0
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["fish"] 