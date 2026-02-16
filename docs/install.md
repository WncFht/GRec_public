# 环境安装指南

本项目依赖 CUDA / torch / deepspeed /（可选）flash-attn 等组件，建议优先使用 **Conda** 或 **Docker** 搭建环境。

## 推荐环境

- Python：`3.10`（更容易与 `flash-attn` / `deepspeed` / 各类多模态模型依赖对齐）
- CUDA：按你的机器/驱动选择（`torch` 版本需要与 CUDA 对齐）
- GPU：多模态模型训练/抽 embedding 强烈建议使用 GPU

> 仓库内 `requirements.txt` 含有 `--extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple`（清华镜像）。如你在海外网络环境，可按需删除这一行或改为企业/自建镜像。

---

## 方式 A：Conda + pip（推荐）

```bash
conda create -n grec python=3.10 -y
conda activate grec
python -m pip install -U pip
```

1) 安装 PyTorch（示例：CUDA 12.4）

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

2) 安装项目依赖

```bash
pip install -r requirements.txt
```

3)（可选）安装 RL 相关依赖

```bash
pip install trl bitsandbytes
```

4) 验证

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; import deepspeed; print('ok')"
```

---

## 方式 B：Docker（可复现）

项目提供 `Dockerfile` 与 `docker-compose.yml`。在已安装 NVIDIA Container Toolkit 的机器上：

```bash
docker compose build
docker compose up -d
docker exec -it unifymmgrec-container fish
```

> 容器默认工作目录为 `/app`，并把仓库挂载到容器内同路径。

---

## 安装/运行小贴士

- `flash-attn`：对 `torch/cuda/gcc` 版本较敏感；装不上时可先跳过（`requirements.txt` 中默认未强依赖）。
- `deepspeed`：首次安装可能会编译扩展，耗时较长；建议在稳定的编译工具链环境下安装（Docker/Conda 更稳）。
- `scripts/`：大多数 `.sh` 脚本是“可运行模板”，里面可能包含作者本地路径/环境变量；首次使用请把 `DATA_PATH/CKPT_PATH/BASE_MODEL` 等路径改成你自己的。
