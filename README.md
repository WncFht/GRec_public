# GRec 项目总览

GRec 聚焦多模态生成式推荐，核心流程为 **离散索引构建（SID） → 指导式微调（SFT） → 评测与回归测试**。本 README 整合 `docs/` 目录的说明，突出完整流水线，方便快速落地实验。

## 环境准备

- `setup.sh`: 使用 Conda 建立并安装基础依赖。
- `Dockerfile`: 预置 CUDA/PyTorch 环境，可配合 `docker-compose.yml` 复现容器化部署。
- `requirements.txt`、`pyproject.toml`: 记录 Python 依赖，若手动创建环境请按需安装。

## 数据与预处理（概览）

`docs/dataprocess_readme.md` 记录了数据增强与表征提取脚本的要点：

- `data_process/` 下提供文本增强、图片下载、多模态向量提取等工具（推荐使用 `qwen_embeddings.py`）。
- `scripts/extract_rep.py` 可批量生成多模态表示，为后续 SID 阶段的 `.npy` 输入做准备。

## 流程总览：SID → SFT → Test

### 1. SID：生成离散索引

参考 `docs/sid_readme.md` 与 `index/README.md`。

1. **准备输入向量**：确保 `./data/{DATASET}/{DATASET}.emb-*.npy` 可用，可来自数据预处理阶段的多模态向量。
2. **训练 RQVAE**（构建索引编码器）：
	- 快速启动：编辑 `index/run.sh` 或 `index/Kmeans_test.sh`，设置 `DATASET`、`MODEL_NAME`、`DATA_PATH`、`KMEANS_MODE` 等变量。
	- 核心 Python 接口：`python3 index/main.py --data_path ... --ckpt_dir ... --num_emb_list ... --layers ...`。
	- 关键参数：
	  - 优化相关：`--lr`、`--epochs`、`--batch_size`、`--weight_decay`、`--warmup_epochs`、`--lr_scheduler_type`。
	  - 模型结构：`--layers`（编码器/解码器宽度）、`--e_dim`（潜空间维度）、`--num_emb_list`（残差量化层及码本大小）。
	  - 量化设置：`--quant_loss_weight`、`--beta`、`--loss_type`、`--kmeans_init`、`--large_scale_kmeans`、`--kmeans_iters`、`--sk_epsilons`、`--sk_iters`。
	  - 输出与日志：`--ckpt_dir`、`--save_limit`、`--use_wandb`、`--device`。
3. **生成索引 JSON**：
	- 使用 `index/generate.sh` 或直接运行 `python3 index/generate_indices.py --dataset ... --ckpt_path ... --output_dir ... --output_file ... --device cuda:0 --batch_size 64`。
	- 输出示例：`./data/{DATASET}/index/{MODEL_NAME}/{DATASET}.index_{MODEL_NAME}.json`，格式为 `{item_id: ["<a_0>", "<b_12>", ...]}`。
	- 若存在冲突，脚本会自动启用 Sinkhorn-Knopp 迭代缓解。
4. **可选评估**：`python3 index/evaluate_index.py --ckpt_path ... --batch_size 2048`，关注碰撞率与各层码本利用率。

生成的索引文件将在 SFT 阶段通过 `--index_file` 使用，是后续任务的离散化桥梁。

### 2. SFT：多任务监督微调

详见 `docs/finetune_readme.md` 与 `scripts/finetune/`。

1. **脚本选择**：
	- `train_ddp_vl.py`: 多模态 VLM（会添加新 token）。
	- `train_ddp_vl_nonewtoken.py`: 不新增 token。
	- `train_ddp.py`: 纯 LLM（如 Qwen2.5、LLaMA）。
	- `train_muon.py`: Muon 优化器
2. **多卡训练示例**（VL + LoRA，新索引用于推荐任务）：

	```bash
	nohup torchrun --nproc_per_node=4 --master_port=33325 -m src.finetune.train_ddp_vl \
	  --seed 42 \
	  --base_model $BASE_MODEL \
	  --model_type $MODEL_TYPE \
	  --output_dir $OUTPUT_DIR \
	  --dataset $DATASET \
	  --data_path $DATA_PATH \
	  --per_device_batch_size 12 \
	  --gradient_accumulation_steps 2 \
	  --use_gradient_checkpointing \
	  --num_workers 32 \
	  --learning_rate 5e-5 \
	  --epochs 4 \
	  --weight_decay 0.01 \
	  --save_and_eval_strategy epoch \
	  --deepspeed ./config/ds_z2_bf16.json \
	  --bf16 \
	  --use_lora \
	  --lora_modules_to_save "embed_tokens,lm_head" \
	  --only_train_response \
	  --tasks item2index,seqrec,fusionseqrec \
	  --train_prompt_sample_num 1,1,1 \
	  --train_data_sample_num 0,0,0 \
	  --ratio_dataset 1 \
	  --report_to wandb \
	  --index_file ./data/$DATASET/index/$MODEL_NAME/${DATASET}.index_${MODEL_NAME}.json
	```

	- 批量大小 = `nproc_per_node * per_device_batch_size * gradient_accumulation_steps`。
	- `--tasks` 对应多任务训练（如 `item2index`, `seqrec`, `fusionseqrec`等）。
	- `--index_file` 指向 SID 阶段产出的离散索引。
	- LoRA 仅更新注意力与 FFN，`lora_modules_to_save` 可额外保留嵌入或 LM Head 全量参数。
3. **产出物**：LoRA 适配器、额外 token embedding。若使用 Zero3，请参考脚本注释在训练结束后执行合并。

### 3. Test：序列推荐与文本生成评测

参考 `docs/test_readme.md` 与 `scripts/seqrec/`、`scripts/text_generate/`。

1. **序列推荐 / Fusion 评测**：
	- 主要脚本：`scripts/seqrec/case_seqrec.sh`、`scripts/seqrec/metric_ddp.sh`。
	- 核心参数：`--test_task`（`seqrec`、`fusionseqrec`、`item2index` 等）、`--lora`、`--base_model`、`--ckpt_model`。
	- LoRA 推理需提供基座模型与 LoRA 权重；纯全量模型仅需 `--ckpt_model`。
	- `metric_ddp.sh` 支持多卡评测并改进结果落盘逻辑。
2. **文本生成任务**：
	- 入口在 `scripts/text_generate/`，无 LoRA 使用 `evaluate*.sh`，LoRA 使用 `evaluate_lora.sh`。
	- 关注 text_enrich (Task9) 任务，指标包括 BLEU、ROUGE 等（参见 `text_generation/evaluate.py`）。
3. **测试数据与基线**：
	- 序列推荐基线：TIGER、LC-Rec（使用文本向量），以及项目内离散化后的多模态向量。
	- 文本生成基线：原始 Qwen-VL、BLIP2、InstructBLIP 等（加载即用）。

## 附加资源

- `docs/notebook_readme.md`: 说明 `notebook/` 内的探索性分析，包含数据洞察、Embedding 对比、Beam Search 结果等。
- 具体可以参考 `docs` 下的各个文档

