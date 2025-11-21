import argparse
import math
import os
import sys
from dataclasses import asdict

import torch
from datasets import Dataset as HFDataset
from trl import GRPOConfig

# --- 本地模块导入 ---
# 1. 导入 Trainer 和 Reward 模型
from minionerec_trainer import ReReTrainer
from sasrec import SASRec

# 2. 导入 Dataset 定义 (使用 data_rl.py 中带 to_verl_records 的版本)
from data_rl import SeqRecDataset, FusionSeqRecDataset

# 3. 导入参数解析和类型定义
import parser as args_parser
from type import (
    Args,
    GlobalArgs,
    DatasetArgs,
    TrainingArgs,
    TestArgs,
    TextGenerationArgs,
)

# 4. 导入工具函数 (这是本次优化的核心)
from utils import (
    load_model_for_training,
    set_seed,
    make_run_name,
    ensure_dir
)

# Mock prompt imports if missing
try:
    from prompt import all_prompt
except ImportError:
    all_prompt = {}


def get_grouped_args(parsed_args: argparse.Namespace) -> Args:
    """将扁平的 argparse.Namespace 转换为结构化的 Args Dataclass"""
    arg_dict = vars(parsed_args)

    def create_from_dict(dataclass_type, source):
        keys = dataclass_type.__annotations__.keys()
        filtered = {k: v for k, v in source.items() if k in keys}
        return dataclass_type(**filtered)

    return Args(
        global_args=create_from_dict(GlobalArgs, arg_dict),
        dataset_args=create_from_dict(DatasetArgs, arg_dict),
        train_args=create_from_dict(TrainingArgs, arg_dict),
        test_args=create_from_dict(TestArgs, arg_dict),
        text_generation_args=create_from_dict(TextGenerationArgs, arg_dict),
    )


def main():
    # ====================================================
    # 1. 参数解析 (使用 parser.py)
    # ====================================================
    parser = argparse.ArgumentParser()
    parser = args_parser.parse_global_args(parser)
    parser = args_parser.parse_dataset_args(parser)
    parser = args_parser.parse_train_args(parser)
    parser = args_parser.parse_test_args(parser)
    
    parsed_args = parser.parse_args()  # 扁平对象，传给 utils.* 使用
    cfg = get_grouped_args(parsed_args) # 结构化对象，传给 Trainer/Dataset 使用

    # ====================================================
    # 2. 环境设置 (使用 utils.py)
    # ====================================================
    # 生成 Run Name
    run_name = make_run_name(parsed_args)
    parsed_args.run_name = run_name # 回写到 args 以供 utils 内部使用
    cfg.train_args.run_name = run_name

    # 设置 WANDB
    if parsed_args.run_name and parsed_args.run_name != "none":
        os.environ["WANDB_PROJECT"] = "rl_rec" 
        os.environ["WANDB_RUN_NAME"] = run_name
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # 设置随机种子
    set_seed(parsed_args.seed)
    ensure_dir(parsed_args.output_dir)

    print(f"Run Name: {run_name}")
    print(f"Model Type: {parsed_args.model_type}")
    print(f"Base Model: {parsed_args.base_model}")

    # ====================================================
    # 3. 数据集准备
    # ====================================================
    # 注意：我们需要先加载 Dataset 以便获取 new_tokens (用于扩展词表)
    
    # 根据任务选择 Dataset 类
    if "fusionseqrec" in parsed_args.tasks:
        DatasetClass = FusionSeqRecDataset
        print("Using FusionSeqRecDataset (Multimodal/Enriched)")
    else:
        DatasetClass = SeqRecDataset
        print("Using SeqRecDataset (Standard ID-based)")

    # 解析采样数 (适配 utils 中的逗号分隔逻辑)
    try:
        sample_list = parsed_args.train_data_sample_num.split(',')
        # 假设 tasks="seqrec,..."，取对应位置的 sample num
        # 这里简化处理，取列表中的最大值或者特定索引
        sample_num = int(sample_list[3]) if len(sample_list) > 3 else int(sample_list[0])
    except:
        sample_num = 0

    print("Processing Train Dataset...")
    # 实例化 Dataset (使用 data_rl.py 的逻辑)
    raw_train_ds = DatasetClass(
        args=parsed_args,  # 传入 Namespace，BaseDataset 会处理
        dataset=parsed_args.dataset,
        mode="train",
        sample_num=sample_num
    )
    
    # 获取新 Token (用于 resize embeddings)
    new_tokens = raw_train_ds.get_new_tokens()
    print(f"Found {len(new_tokens)} new tokens from dataset.")

    # 转换为 RL 格式 (to_verl_records)
    train_records = raw_train_ds.to_verl_records("train")
    train_dataset = HFDataset.from_list(train_records)
    train_dataset = train_dataset.shuffle(seed=parsed_args.seed)

    print("Processing Eval Dataset...")
    raw_eval_ds = DatasetClass(
        args=parsed_args,
        dataset=parsed_args.dataset,
        mode="valid",
        prompt_id=parsed_args.valid_prompt_id
    )
    eval_records = raw_eval_ds.to_verl_records("valid")
    eval_dataset = HFDataset.from_list(eval_records)

    print(f"Train Size: {len(train_dataset)}")
    print(f"Eval Size: {len(eval_dataset)}")

    # ====================================================
    # 4. 模型加载 (使用 utils.load_model_for_training)
    # ====================================================
    # 这个函数封装了: Tokenizer, Resize Embeddings, LoRA, Freeze
    
    model, processor, orig_vocab, new_vocab, _, embedding_hooks = load_model_for_training(
        args=parsed_args,
        new_tokens=new_tokens,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        logger=None # 使用 print
    )

    # 从 processor 获取 tokenizer
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    
    # 确保 pad_token 存在 (GRPO 必须)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 某些模型可能需要手动设置 pad_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

    # ====================================================
    # 5. 构建 Reward Functions
    # ====================================================
    # 建立查找表 (闭包用)
    prompt2target = {}
    prompt2history = {}
    
    all_records = train_records + eval_records
    for row in all_records:
        p = row['prompt']
        if isinstance(p, list): p = p[0]['content']
        gt = row['reward_model']['ground_truth']
        prompt2target[p] = gt
        prompt2history[p] = row['extra_info'].get('inters', "")

    # 定义 Rule Reward
    def get_gt(prompt):
        return prompt2target.get(prompt)

    def rule_reward(prompts, completions, **kwargs):
        rewards = []
        for p, c in zip(prompts, completions):
            gt = get_gt(p)
            if gt is None:
                rewards.append(0.0)
                continue
            # 简单清洗
            if c.strip('\n" ') == gt.strip('\n" '):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    # SASRec Reward (可选)
    reward_funcs = [rule_reward]
    
    # 简单的 SASRec 激活判断逻辑
    if "sasrec" in parsed_args.tasks.lower() or cfg.test_args.ckpt_path: 
        print("Attempting to load SASRec for reward...")
        sasrec_ckpt = cfg.test_args.ckpt_path
        if os.path.exists(sasrec_ckpt):
            item_num = len(raw_train_ds.get_all_items())
            sasrec_model = SASRec(32, item_num, parsed_args.max_his_len, 0.3, parsed_args.device)
            sasrec_model.load_state_dict(torch.load(sasrec_ckpt))
            sasrec_model.to(parsed_args.device)
            sasrec_model.eval()
            
            def cf_reward(prompts, completions, **kwargs):
                # TODO: 实现 Token Str -> ID 的转换逻辑
                return [0.0] * len(completions)
            
            reward_funcs.append(cf_reward)

    # ====================================================
    # 6. 配置 Trainer
    # ====================================================
    # 映射参数到 GRPOConfig
    training_args = GRPOConfig(
        output_dir=parsed_args.output_dir,
        run_name=run_name,
        learning_rate=parsed_args.learning_rate,
        per_device_train_batch_size=parsed_args.per_device_batch_size,
        per_device_eval_batch_size=parsed_args.per_device_batch_size if hasattr(parsed_args, 'per_device_batch_size') else 1,
        gradient_accumulation_steps=parsed_args.gradient_accumulation_steps,
        num_train_epochs=parsed_args.epochs,
        bf16=parsed_args.bf16,
        fp16=parsed_args.fp16,
        optim=parsed_args.optim,
        warmup_ratio=parsed_args.warmup_ratio,
        save_strategy=parsed_args.save_and_eval_strategy,
        save_steps=parsed_args.save_and_eval_steps,
        logging_steps=parsed_args.logging_step,
        report_to="wandb" if parsed_args.run_name != "none" else "none",
        
        # GRPO 特定参数
        num_generations=parsed_args.num_beams if hasattr(parsed_args, 'num_beams') else 8, # 复用 num_beams 或默认
        max_completion_length=parsed_args.max_new_tokens,
        max_prompt_length=parsed_args.model_max_length,
        temperature=parsed_args.temperature if hasattr(parsed_args, 'temperature') else 1.0,
        use_vllm=False, 
    )

    # 约束生成设置 (Trie)
    prefix_allowed_tokens_fn = None
    if parsed_args.use_constrained_generation:
        print("Building Trie for Constrained Generation...")
        prefix_allowed_tokens_fn = raw_train_ds.get_prefix_allowed_tokens_fn(tokenizer)

    # 初始化自定义 Trainer
    trainer = ReReTrainer(
        model=model,           # 直接传入 utils 加载好的模型 (可能是 PeftModel)
        base_model=parsed_args.base_model, 
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prompt2history=prompt2history, # 兼容旧代码
        history2target={},             # 兼容旧代码
    )

    if prefix_allowed_tokens_fn:
        trainer.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    # ====================================================
    # 7. 训练与保存
    # ====================================================
    print("Starting Training...")
    trainer.train()

    print(f"Saving model to {parsed_args.output_dir}")
    trainer.save_model(parsed_args.output_dir)
    
    # 保存最终 checkpoint
    final_dir = os.path.join(parsed_args.output_dir, "final_checkpoint")
    ensure_dir(final_dir)
    
    # 处理 LoRA 或 Full Model 保存
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # 保存 token metadata
    if hasattr(model, "config"):
        model.config.save_pretrained(final_dir)

    print("Training Finished.")

if __name__ == "__main__":
    main()