#!/bin/bash

# --- Configuration ---
# 定义镜像名称和容器名称，方便管理
IMAGE_NAME="docker.v2.aispeech.com/sjtu/sjtu_yukai-fanghaotian:latest"

# --- Step 1: Build the Docker Image ---
# 使用当前目录的 Dockerfile 构建镜像，并打上指定的标签
echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .
echo "Image build complete."
echo ""


# --- Step 2: Run the Docker Container ---
# 使用 'docker run' 命令启动容器，并模拟 docker-compose.yml 的配置
#
# 参数解释:
# -d                : 在后台以分离模式运行
# --rm              : 容器停止后自动删除，方便清理
# --name            : 指定容器的名称
# --gpus all        : 将所有可用的 GPU 挂载到容器中
# --shm-size=64g    : 设置共享内存大小为 64GB，以避免 DataLoader 错误
# -v "$(pwd)":/app : 将当前目录挂载到容器的 /app 目录
# -e WANDB_API_KEY  : 传递 WANDB API Key 环境变量
# -it               : 保持标准输入打开并分配一个伪TTY，实现交互式会话
# $IMAGE_NAME       : 指定要运行的镜像
#
# 注意:
# 1. 运行此命令前，请确保已经停止并移除了同名的旧容器，
#    可以使用 'docker stop unifymmgrec-container && docker rm unifymmgrec-container'
# 2. 如果 WANDB_API_KEY 环境变量未设置，请在命令中直接提供或提前 export。
echo "To run the container, use the following command:"
echo "----------------------------------------------------"
echo "docker run -d --rm \\"
echo "  --gpus all \\"
echo "  --shm-size=64g \\"
echo "  -v \"\$(pwd)\":/app \\"
echo "  -e WANDB_API_KEY=\${WANDB_API_KEY} \\"
echo "  -it \"$IMAGE_NAME\""
echo "----------------------------------------------------"
echo ""
echo "Example command to stop and remove a running container before starting a new one:"
echo ""
echo "Example command to attach to the running container:"
echo "docker exec -it $CONTAINER_NAME fish"

docker run -d --rm \
  --gpus all \
  --shm-size=64g \
  -v "$(pwd)":/app \
  -e WANDB_API_KEY=\${WANDB_API_KEY} \
  -it "docker.v2.aispeech.com/sjtu/sjtu_yukai-fanghaotian:latest"