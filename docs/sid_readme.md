# Index 模块说明

> 注意所有脚本要在项目根目录下进行

该目录实现了使用残差向量量化自编码器（RQVAE）对物品向量做压缩建索引的完整流程，包含训练、索引生成与指标评估脚本。数据默认存放在 `./data/{DATASET}/`，模型权重输出在 `./data/{DATASET}/index/{MODEL_NAME}/`。

- 训练入口：`main.py`（可直接调用或通过 `run.sh` / `Kmeans_test.sh` 包装脚本）
- 索引生成：`generate_indices.py`（可通过 `generate.sh` 调用）
- 评估指标：`evaluate_index.py`（可通过 `evaluate.sh` 调用）

## 数据准备

- `datasets.EmbDataset` 期望输入为 `.npy` 格式的二维数组，形状为 `[num_items, dim]`。
- `--data_path` 指向该文件路径；训练脚本会根据数据文件推断输入维度。

## 训练 (`main.py`)

核心参数（括号内为默认值）：

- `--data_path` (`../data/Games/Games.emb-llama-td.npy`): 训练用的嵌入文件。
- `--ckpt_dir` (空): 模型检查点输出目录，会在其下创建时间戳文件夹。
- `--device` (`cuda:0`): 训练设备。
- `--lr`, `--weight_decay`, `--epochs`, `--batch_size`, `--num_workers`, `--eval_step`: 常规优化参数。
- `--learner` (`AdamW`), `--lr_scheduler_type` (`constant`), `--warmup_epochs` (`50`): 优化器及学习率调度策略。
- `--layers` (`2048 1024 512 256 128 64`), `--e_dim` (`32`): 编码器/解码器 MLP 结构及低维潜空间大小。
- `--num_emb_list` (`256 256 256`): 每个量化层的码本大小；列表长度决定残差量化层数。
- `--quant_loss_weight` (`1.0`), `--beta` (`0.25`), `--loss_type` (`mse`): 损失函数配置。
- `--kmeans_init` (`True|False`), `--large_scale_kmeans` (`True|False`), `--kmeans_iters` (`100`): 控制码本初始化方式。
- `--sk_epsilons` (`0.0 0.0 0.0`), `--sk_iters` (`50`): Sinkhorn-Knopp 参数，决定冲突消解时的软分配强度。
- `--save_limit` (`5`): 保留的最近模型数量，脚本会自动清理旧检查点。
- `--use_wandb`, `--wandb_project`, `--wandb_name`: 可选的 Weights & Biases 记录。

日志和模型文件默认写入 `./log/index/` 和 `--ckpt_dir` 指定目录。

### 快速启动脚本

- `run.sh`: 默认训练脚本，设置 `DATASET`、`MODEL_NAME`、`DATA_PATH`、`KMEANS_MODE` 等变量后执行。脚本使用 `nohup` 后台训练，日志输出到 `./log/index/index_YYYYMMDDHHMMSS.log`。
- `Kmeans_test.sh`: 针对 K-Means 初始化的 A/B 测试脚本，通过 `KMEANS_MODE` 在 `large` / `small` / `none` 三种配置间切换，并自动拼接 WandB 运行名称。

如需自定义参数，可直接调用：

```bash
python3 index/main.py \
	--data_path ./data/Instruments/Instruments.emb-llama-td.npy \
	--ckpt_dir ./data/Instruments/index/llama/ \
	--num_emb_list 256 256 256 256 \
	--layers 2048 1024 512 256 128 64 \
	--kmeans_init True --large_scale_kmeans True
```

## 索引生成 (`generate_indices.py`)

该脚本从训练得到的检查点生成可部署的离散索引，并在检测到碰撞时按需启用 Sinkhorn-Knopp 以迭代消解。输出 JSON 形如 `{item_id: ["<a_0>", "<b_12>", ...]}`。

命令行参数：

- `--dataset`: 数据集名称，仅用于日志和输出文件命名。
- `--ckpt_path`: 训练得到的模型检查点路径（`.pth`）。脚本会读取其中保存的 `args` 来恢复数据配置。
- `--output_dir` (`./data`): 结果保存目录，若不存在会自动创建。
- `--output_file`: 输出文件名，例如 `Instruments.index_llama.json`。
- `--device` (`cuda:0`): 推理设备。
- `--batch_size` (`64`): 推理批大小。

辅助脚本 `generate.sh` 演示了常见调用方式，请按需修改 `CKPT_PATH`、`OUTPUT_DIR`、`OUTPUT_FILE` 等变量后执行。

## 模型评估 (`evaluate_index.py`)

用于离线评估训练好的检查点，输出碰撞率及各量化层的码本利用率。

参数说明：

- `--ckpt_path`: 需要评估的模型权重文件。
- `--device` (`cuda:0`): 推理设备。
- `--batch_size` (`2048`): 评估批大小。

`evaluate.sh` 提供了可直接运行的示例，只需更新 `DATASET`、`MODEL_NAME`、`TIMESTAMP` 和 `MODEL_FILE` 与实际目录一致即可。

## 其它文件

- `trainer.py`: 封装训练循环，负责保存多种最优模型（最低损失、最低碰撞率、最高码本利用率），并在 `--large_scale_kmeans` 为真时执行一次性大规模 K-Means 初始化。
- `models/`: 包含 RQVAE 主体、残差量化器与支持的 K-Means、Sinkhorn 实现。
- `utils.py`: 提供日志着色、目录创建和文件清理等通用工具。

请根据具体实验需求调整脚本中的路径与参数，运行前确保所需数据及依赖已就绪。