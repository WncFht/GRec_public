import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
from transformers import TrainingArguments

from ..collator import MultiModalCollator
from ..logger import (
    configure_tqdm_for_file_output,
    get_tqdm_compatible_logger,
    log_args,
)
from ..parser import parse_dataset_args, parse_global_args, parse_train_args
from ..utils import (
    ensure_dir,
    load_datasets,
    load_model_for_training,
    make_run_name,
    set_seed,
)


class UnifiedTrainer:
    """统一的训练器类，处理全量微调和LoRA微调"""

    def __init__(self, args: argparse.Namespace):
        """
        初始化训练器

        Args:
            args: 训练参数

        """
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = self.world_size != 1

        # 生成run_name
        self.args.run_name = make_run_name(self.args)

        # 初始化logger
        self._init_logger()

        # 设置环境
        self._setup_environment()

    def _init_logger(self):
        """初始化日志记录器"""
        debug_mode = self.args.debug if hasattr(self.args, "debug") else False

        self.logger = get_tqdm_compatible_logger(
            name="unified_multitask_train",
            output_dir=self.args.output_dir,
            run_name=self.args.run_name,
            debug=debug_mode,
            rank=self.local_rank,
        )

        # 配置tqdm
        configure_tqdm_for_file_output(use_file_output=not debug_mode)

        # 记录训练模式
        train_mode = (
            "LoRA finetuning" if self.args.use_lora else "Full finetuning"
        )
        self.logger.info(f"Starting multitask {train_mode}")
        self.logger.info(f"RUN_NAME: {self.args.run_name}")

    def _setup_environment(self):
        """设置训练环境"""
        set_seed(self.args.seed)
        ensure_dir(self.args.output_dir)

        if self.ddp:
            torch.cuda.set_device(self.local_rank)
            self.device_map = {"": self.local_rank}
        else:
            self.device_map = "auto"

        if self.local_rank == 0:
            train_mode = "LoRA" if self.args.use_lora else "Full"
            self.logger.info(f"Training mode: {train_mode} finetuning")
            self.logger.info(f"DDP: {self.ddp} (World size: {self.world_size})")
            self.logger.info(f"Device map: {self.device_map}")
            log_args(self.logger, self.args, "Training Configuration")

    def _get_training_args(self) -> TrainingArguments:
        """构建训练参数"""
        # 根据配置选择报告工具
        if hasattr(self.args, "report_to"):
            report_to = self.args.report_to
        else:
            report_to = "tensorboard" if self.args.use_lora else "wandb"

        self.logger.info(f"Report to: {report_to}")

        return TrainingArguments(
            seed=self.args.seed,
            per_device_train_batch_size=self.args.per_device_batch_size,
            per_device_eval_batch_size=self.args.per_device_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_ratio=self.args.warmup_ratio,
            num_train_epochs=self.args.epochs,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            lr_scheduler_type=self.args.lr_scheduler_type,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            logging_steps=self.args.logging_step,
            optim=self.args.optim,
            gradient_checkpointing=self.args.use_gradient_checkpointing,
            eval_strategy=self.args.save_and_eval_strategy,
            save_strategy=self.args.save_and_eval_strategy,
            eval_steps=self.args.save_and_eval_steps,
            save_steps=self.args.save_and_eval_steps,
            output_dir=self.args.output_dir,
            load_best_model_at_end=True,
            deepspeed=self.args.deepspeed,
            ddp_find_unused_parameters=False if self.ddp else None,
            dataloader_num_workers=self.args.num_workers,
            dataloader_pin_memory=True,
            accelerator_config={"non_blocking": True},
            remove_unused_columns=False,
            report_to=report_to,
            run_name=self.args.run_name,
            eval_delay=1
            if self.args.save_and_eval_strategy == "epoch"
            else 2000,
        )

    def _load_model_and_data(self) -> tuple:
        """加载模型和数据集"""
        # 使用统一的load_model_for_training函数
        (
            model,
            processor,
            original_vocab_size,
            new_vocab_size,
            new_tokens,
            embedding_hooks,
        ) = load_model_for_training(
            self.args,
            new_tokens=None,
            local_rank=self.local_rank,
            logger=self.logger,
            nonewtokens=True,
        )

        # 加载数据集
        train_data, valid_data = load_datasets(
            self.args, self.logger, self.local_rank
        )

        # 记录统计信息
        self._log_statistics(train_data)

        # 保存processor和config
        if self.local_rank == 0:
            self._save_configs(processor)

        return (
            model,
            processor,
            train_data,
            valid_data,
            embedding_hooks,
        )

    def _log_statistics(
        self,
        train_data,
    ):
        """记录训练统计信息"""
        if self.local_rank == 0:
            if self.args.use_lora:
                self.logger.info(
                    f"LoRA config: r={self.args.lora_r}, alpha={self.args.lora_alpha}, dropout={self.args.lora_dropout}"
                )
                self.logger.info(
                    f"Target modules: {self.args.lora_target_modules}"
                )
                if (
                    hasattr(self.args, "lora_modules_to_save")
                    and self.args.lora_modules_to_save
                ):
                    self.logger.info(
                        f"Modules to save: {self.args.lora_modules_to_save}"
                    )

            self.logger.info(f"Train samples: {len(train_data)}")

            # 计算有效batch size
            if (
                hasattr(self.args, "use_gradient_checkpointing")
                and self.args.use_gradient_checkpointing
            ):
                effective_batch_size = (
                    self.args.per_device_batch_size
                    * self.args.gradient_accumulation_steps
                    * self.world_size
                )
            else:
                effective_batch_size = (
                    self.args.per_device_batch_size * self.world_size
                )

            self.logger.info(f"Effective batch size: {effective_batch_size}")
            self.logger.info(
                f"Steps per epoch: {len(train_data) / self.args.per_device_batch_size:.2f}"
            )

    def _save_configs(self, processor) -> None:
        """
        保存 processor 与模型 config，
        仅当确定某字段为 vocab_size 且与旧值不同时才更新，保证鲁棒。
        """
        import os

        from transformers import AutoConfig

        out_dir = self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1. 保存 processor
        processor.save_pretrained(out_dir)

        # 2. 加载 config
        config = AutoConfig.from_pretrained(
            self.args.base_model, trust_remote_code=True
        )

        # 4. 保存修改后的 config
        config.save_pretrained(out_dir)
        self.logger.info(f"Saved processor and config to {out_dir}")

    def train(self):
        """执行训练流程"""
        # 加载模型和数据
        (
            model,
            processor,
            train_data,
            valid_data,
            embedding_hooks,
        ) = self._load_model_and_data()

        # 如果是LoRA模式，打印可训练参数
        if self.args.use_lora and self.local_rank == 0:
            model.print_trainable_parameters()

        # 创建数据collator
        collator = MultiModalCollator(self.args, processor)

        # 设置模型并行（如果需要）
        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        # 获取训练参数
        training_args = self._get_training_args()

        # 创建trainer
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=valid_data,
            args=training_args,
            processing_class=processor,
            data_collator=collator,
        )

        # 编译模型（如果支持）
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model)

        # 开始训练
        self.logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=self.args.resume_from_checkpoint)
        self.logger.info("Training completed.")

        # 清理embedding梯度hook
        if embedding_hooks:
            for hook in embedding_hooks:
                hook.remove()
            self.logger.info(
                f"Removed {len(embedding_hooks)} embedding gradient hooks"
            )

        # 保存模型和状态
        self.logger.info("Saving model and training state...")
        trainer.save_state()
        trainer.save_model(output_dir=self.args.output_dir)
        self.logger.info(f"Model and state saved to {self.args.output_dir}")

        self.logger.info("Training pipeline completed successfully!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一的多任务训练脚本")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)  # 已包含所有LoRA参数

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        choices=["wandb", "tensorboard", "none"],
        help="Reporting tool (default: tensorboard for LoRA, wandb for full finetuning)",
    )

    args = parser.parse_args()

    # 创建训练器并执行训练
    trainer = UnifiedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
