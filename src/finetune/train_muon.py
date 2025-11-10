"""
统一的多任务训练脚本，支持全量微调和LoRA微调两种模式。

使用方法：
- 全量微调：python unified_multitask_train.py --base_model xxx ...
- LoRA微调：python unified_multitask_train.py --base_model xxx --use_lora ...
"""

import argparse
import os
import sys
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import os
import pickle

import torch
import transformers
from transformers import (
    TrainingArguments,
)

from src.collator import MultiModalCollator
from src.logger import (
    configure_tqdm_for_file_output,
    get_tqdm_compatible_logger,
    log_args,
)
from src.parser import parse_dataset_args, parse_global_args, parse_train_args
from src.utils import (
    ensure_dir,
    load_datasets,
    load_model_for_training,
    make_run_name,
    set_seed,
)


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.

    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [
                p for p in group["params"] if not self.state[p]["use_muon"]
            ]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


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
            ddp_find_unused_parameters=False if self.ddp else None,
            dataloader_num_workers=self.args.num_workers,
            remove_unused_columns=False,
            report_to=report_to,
            run_name=self.args.run_name,
            eval_delay=1
            if self.args.save_and_eval_strategy == "epoch"
            else 2000,
            optim_target_modules=None,
            optim_args=None,
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
        )

        # 加载数据集
        train_data, valid_data = load_datasets(self.args)

        # 记录统计信息
        self._log_statistics(train_data, original_vocab_size, new_vocab_size)

        # 保存processor和config
        if self.local_rank == 0:
            self._save_configs(processor, new_vocab_size)

        return (
            model,
            processor,
            train_data,
            valid_data,
            embedding_hooks,
            original_vocab_size,
            new_vocab_size,
            new_tokens,
        )

    def _log_statistics(self, train_data, original_vocab_size, new_vocab_size):
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

            self.logger.info(
                f"Added {new_vocab_size - original_vocab_size} new tokens"
            )
            self.logger.info(f"Original vocab size: {original_vocab_size}")
            self.logger.info(f"New vocab size: {new_vocab_size}")
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

    def _save_configs(self, processor, new_vocab_size):
        """保存配置文件"""
        processor.save_pretrained(self.args.output_dir)

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.base_model, trust_remote_code=True
        )
        config.vocab_size = new_vocab_size
        config.save_pretrained(self.args.output_dir)

        self.logger.info(
            f"Saved processor and config to {self.args.output_dir}"
        )

    def _save_new_token_embeddings(
        self,
        model: Any,
        original_vocab_size: int,
        new_vocab_size: int,
        new_tokens: list[str],
    ) -> None:
        """保存新token的embedding用于后续可视化"""
        # 对于LoRA模型，需要从base model获取embedding
        if hasattr(model, "get_base_model"):
            base_model = model.get_base_model()
        else:
            base_model = model

        new_token_embeddings = (
            base_model.get_input_embeddings()
            .weight[original_vocab_size:]
            .detach()
            .cpu()
        )

        # 转换为 float32 类型，避免 BFloat16 的兼容性问题
        new_token_embeddings = new_token_embeddings.float()

        # 保存 embedding 和 token 名称
        embedding_info = {
            "embeddings": new_token_embeddings,
            "token_names": new_tokens,
            "original_vocab_size": original_vocab_size,
            "new_vocab_size": new_vocab_size,
        }

        embedding_save_path = os.path.join(
            self.args.output_dir, "new_token_embeddings.pkl"
        )
        with open(embedding_save_path, "wb") as f:
            pickle.dump(embedding_info, f)

        self.logger.info(
            f"New token embeddings saved to: {embedding_save_path}"
        )

    def train(self):
        """执行训练流程"""
        # 加载模型和数据
        (
            model,
            processor,
            train_data,
            valid_data,
            embedding_hooks,
            original_vocab_size,
            new_vocab_size,
            new_tokens,
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

        # ---------------- Muon 优化器 ----------------
        muon_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in n and "lm_head" not in n
        ]
        adamw_params = [
            p
            for n, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in n and "lm_head" not in n
            )
        ]
        optimizer = Muon(
            lr=self.args.learning_rate,
            wd=self.args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                self.args.warmup_ratio
                * len(train_data)
                / self.args.per_device_batch_size
                * self.args.epochs
            ),
            num_training_steps=len(train_data)
            * self.args.epochs
            // self.args.per_device_batch_size,
        )

        # 创建trainer
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=valid_data,
            args=training_args,
            processing_class=processor,
            data_collator=collator,
            optimizers=(optimizer, scheduler),
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

        # 保存新token embeddings
        self._save_new_token_embeddings(
            model,
            original_vocab_size,
            new_vocab_size,
            new_tokens,
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
