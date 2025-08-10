import heapq
import logging
import os
from time import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from utils import delete_file, ensure_dir, get_local_time, set_color

import wandb


class Trainer:
    """
    训练器类，用于管理模型的训练、评估和保存。
    """

    def __init__(self, args, model, data_num):
        """
        初始化训练器。

        参数:
            args: 包含各种配置参数的对象。
            model: 要训练的模型实例。
            data_num: 训练数据集中的样本总数。
        """
        self.args = args
        self.model = model
        self.logger = logging.getLogger()  # 获取日志记录器

        # 是否使用 Weights & Biases 进行实验跟踪
        self.use_wandb = getattr(self.args, "use_wandb", False)
        if self.use_wandb:
            wandb_project = getattr(self.args, "wandb_project", "unifymmgrec")
            wandb_name = getattr(self.args, "wandb_name", None)
            wandb.init(
                project=wandb_project,
                config=self.args,
                reinit=True,
                name=wandb_name,
            )
            # 监控模型参数和梯度，并记录到 WandB
            wandb.watch(
                self.model, log="all", log_freq=max(100, data_num // 10)
            )

        # 优化器和学习率调度器相关参数
        self.lr = args.lr  # 学习率
        self.learner = args.learner  # 优化器类型 (例如: adam, sgd)
        self.lr_scheduler_type = (
            args.lr_scheduler_type
        )  # 学习率调度器类型 (例如: linear, constant)

        self.weight_decay = args.weight_decay  # 权重衰减
        self.epochs = args.epochs  # 训练轮次
        self.warmup_steps = args.warmup_epochs * data_num  # 学习率预热步数
        self.max_steps = args.epochs * data_num  # 总训练步数

        # 模型保存相关参数
        self.save_limit = args.save_limit  # 最多保存的模型数量
        self.best_save_heap = []  # 存储最佳模型的堆，用于管理保存的模型
        self.newest_save_queue = []  # 存储最新保存模型的队列
        self.eval_step = min(
            args.eval_step, self.epochs
        )  # 评估步长，每多少个 epoch 进行一次评估
        self.device = args.device  # 设备 (CPU 或 GPU)
        self.device = torch.device(
            self.device
        )  # 将设备字符串转换为 torch.device 对象
        self.ckpt_dir = args.ckpt_dir  # 检查点保存目录
        saved_model_dir = (
            f"{get_local_time()}"  # 根据当前时间生成保存模型的子目录
        )
        self.ckpt_dir = os.path.join(
            self.ckpt_dir, saved_model_dir
        )  # 完整的检查点保存路径
        ensure_dir(self.ckpt_dir)  # 确保检查点目录存在

        # 最佳损失和碰撞率记录
        self.best_loss = np.inf  # 记录最佳训练损失
        self.best_collision_rate = np.inf  # 记录最佳碰撞率
        self.best_loss_ckpt = "best_loss_model.pth"  # 最佳损失模型的保存文件名
        self.best_collision_ckpt = (
            "best_collision_model.pth"  # 最佳碰撞率模型的保存文件名
        )
        self.best_codebook_utilization = 0.0  # 记录最佳码本利用率
        self.best_utilization_ckpt = (
            "best_utilization_model.pth"  # 最佳码本利用率模型的保存文件名
        )

        self.optimizer = self._build_optimizer()  # 构建优化器
        self.scheduler = self._get_scheduler()  # 获取学习率调度器
        self.model = self.model.to(self.device)  # 将模型移动到指定设备

    def _build_optimizer(self):
        """
        根据配置参数构建优化器。

        返回:
            torch.optim.Optimizer: 构建好的优化器实例。
        """
        params = self.model.parameters()  # 获取模型参数
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            # 对于 Adagrad 优化器，将其内部状态移动到指定设备
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(
                params, lr=learning_rate
            )  # 默认使用 Adam 优化器
        return optimizer

    def _get_scheduler(self):
        """
        根据配置参数获取学习率调度器。

        返回:
            torch.optim.lr_scheduler._LRScheduler: 学习率调度器实例。
        """
        if self.lr_scheduler_type.lower() == "linear":
            # 线性预热和衰减的学习率调度器
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            # 常数学习率调度器，带预热
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.warmup_steps
            )

        return lr_scheduler

    def _check_nan(self, loss):
        """
        检查损失是否为 NaN (非数字)。

        参数:
            loss: 模型的损失值。

        抛出:
            ValueError: 如果损失为 NaN。
        """
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):
        """
        执行一个训练 epoch。

        参数:
            train_data: 训练数据集的 DataLoader。
            epoch_idx: 当前 epoch 的索引。

        返回:
            tuple: (总损失, 总重构损失)。
        """
        self.model.train()  # 设置模型为训练模式

        total_loss = 0
        total_recon_loss = 0
        # 使用 tqdm 显示训练进度条
        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )
        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)  # 将数据移动到指定设备
            self.optimizer.zero_grad()  # 清零梯度
            out, rq_loss, indices = self.model(data)  # 前向传播
            loss, loss_recon = self.model.compute_loss(
                out, rq_loss, xs=data
            )  # 计算损失
            self._check_nan(loss)  # 检查损失是否为 NaN
            loss.backward()  # 反向传播
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()  # 更新模型参数
            self.scheduler.step()  # 更新学习率
            if self.use_wandb:
                # 记录训练步的损失、重构损失和学习率到 WandB
                wandb.log(
                    {
                        "train_step/loss": loss.item(),
                        "train_step/recon_loss": loss_recon.item(),
                        "train_step/lr": self.scheduler.get_last_lr()[0],
                    }
                )
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        """
        执行一个验证 epoch，计算碰撞率和码本利用率。

        参数:
            valid_data: 验证数据集的 DataLoader。

        返回:
            tuple: (碰撞率, 平均码本利用率)。
        """
        self.model.eval()  # 设置模型为评估模式

        # 使用 tqdm 显示评估进度条
        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate   ", "pink"),
        )

        all_indices = []  # 存储所有索引
        num_sample = 0  # 样本总数
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)  # 获取模型的索引
            # 将索引展平并移动到 CPU
            indices = indices.view(-1, indices.shape[-1]).cpu()
            all_indices.append(indices)

        all_indices = torch.cat(all_indices, dim=0).numpy()

        # 计算碰撞率
        indices_set = set()  # 存储唯一索引的集合
        for index in all_indices:
            code = "-".join(
                [str(int(_)) for _ in index]
            )  # 将索引转换为字符串表示
            indices_set.add(code)  # 添加到唯一索引集合中
        collision_rate = (num_sample - len(indices_set)) / num_sample

        # 计算码本利用率
        num_quantizers = all_indices.shape[1]
        num_emb_list = self.model.num_emb_list
        utilizations = []
        self.logger.info("Detailed Codebook Utilization:")
        for i in range(num_quantizers):
            unique_codes = np.unique(all_indices[:, i])
            utilization = len(unique_codes) / num_emb_list[i]
            utilizations.append(utilization)
            self.logger.info(
                f"  Layer {i}: {utilization:.4f} ({len(unique_codes)}/{num_emb_list[i]})"
            )
            if self.use_wandb:
                wandb.log({f"eval/codebook_utilization_layer_{i}": utilization})

        avg_utilization = np.mean(utilizations)

        return collision_rate, avg_utilization

    def _save_checkpoint(
        self, epoch, collision_rate=1, avg_utilization=0, ckpt_file=None
    ):
        """
        保存模型检查点。

        参数:
            epoch: 当前 epoch 的索引。
            collision_rate: 当前评估的碰撞率。
            avg_utilization: 当前评估的平均码本利用率。
            ckpt_file: 可选的检查点文件名。如果未提供，则根据 epoch 和指标生成文件名。

        返回:
            str: 保存的检查点文件路径。
        """
        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(
                self.ckpt_dir,
                "epoch_%d_collision_%.4f_util_%.4f_model.pth"
                % (epoch, collision_rate, avg_utilization),
            )
        )
        state = {
            "args": self.args,  # 保存训练参数
            "epoch": epoch,  # 保存当前 epoch
            "best_loss": self.best_loss,  # 保存最佳损失
            "best_collision_rate": self.best_collision_rate,  # 保存最佳碰撞率
            "best_codebook_utilization": self.best_codebook_utilization,  # 保存最佳利用率
            "state_dict": self.model.state_dict(),  # 保存模型状态字典
            "optimizer": self.optimizer.state_dict(),  # 保存优化器状态字典
        }
        torch.save(state, ckpt_path, pickle_protocol=4)  # 保存检查点

        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

        return ckpt_path

    def _generate_train_loss_output(
        self, epoch_idx, s_time, e_time, loss, recon_loss
    ):
        """
        生成训练损失的输出字符串。

        参数:
            epoch_idx: 当前 epoch 的索引。
            s_time: 训练开始时间。
            e_time: 训练结束时间。
            loss: 总损失。
            recon_loss: 重构损失。

        返回:
            str: 格式化的训练损失输出字符串。
        """
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output += ", "
        train_loss_output += (
            set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        )
        return train_loss_output + "]"

    def fit(self, data):
        """
        开始模型的训练过程。

        参数:
            data: 训练数据集的 DataLoader。

        返回:
            tuple: (最佳损失, 最佳碰撞率)。
        """
        # --- K-Means Initialization Step ---
        # 检查是否需要并可以执行 K-Means 初始化
        if getattr(self.args, "kmeans_init", False) and getattr(
            self.args, "large_scale_kmeans", False
        ):
            # 检查模型内部的初始化状态，避免重复执行
            is_initted = all(vq.initted for vq in self.model.rq.vq_layers)
            if not is_initted:
                self.logger.info(
                    "Performing LARGE SCALE K-Means initialization..."
                )
                # 从 DataLoader 中获取底层数据集，并抽取一个大的固定样本
                # 使用 EmbDataset 的 .embeddings 属性
                init_data_tensors = data.dataset.embeddings[
                    : min(20000, len(data.dataset))
                ]
                init_data = torch.FloatTensor(init_data_tensors).to(self.device)

                # 执行一次前向传播以触发 K-Means 初始化
                # 使用 no_grad 是因为我们只关心初始化，不需要计算梯度
                with torch.no_grad():
                    self.model(init_data)

                self.logger.info("LARGE SCALE K-Means initialization finished.")

        cur_eval_step = 0  # 当前评估步数

        for epoch_idx in range(self.epochs):
            # 训练阶段
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx,
                training_start_time,
                training_end_time,
                train_loss,
                train_recon_loss,
            )
            self.logger.info(train_loss_output)

            # 评估阶段
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate, avg_utilization = self._valid_epoch(
                    data
                )  # 执行验证 epoch

                # 如果当前训练损失优于历史最佳损失，则保存模型
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(
                        epoch=epoch_idx, ckpt_file=self.best_loss_ckpt
                    )

                # 如果当前碰撞率优于历史最佳碰撞率，则保存模型并重置评估步数
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        avg_utilization=avg_utilization,
                        ckpt_file=self.best_collision_ckpt,
                    )
                else:
                    cur_eval_step += 1  # 否则增加评估步数

                # 如果当前码本利用率优于历史最佳，则保存模型
                if avg_utilization > self.best_codebook_utilization:
                    self.best_codebook_utilization = avg_utilization
                    self._save_checkpoint(
                        epoch=epoch_idx,
                        collision_rate=collision_rate,
                        avg_utilization=avg_utilization,
                        ckpt_file=self.best_utilization_ckpt,
                    )

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %.4f, "
                    + set_color("avg_utilization", "blue")
                    + ": %.4f]"
                ) % (
                    epoch_idx,
                    valid_end_time - valid_start_time,
                    collision_rate,
                    avg_utilization,
                )

                self.logger.info(valid_score_output)

                if self.use_wandb:
                    # 记录 epoch 级别的指标到 WandB
                    wandb.log(
                        {
                            "epoch": epoch_idx,
                            "epoch/train_loss": train_loss / len(data),
                            "epoch/train_recon_loss": train_recon_loss
                            / len(data),
                            "eval/collision_rate": collision_rate,
                            "eval/avg_codebook_utilization": avg_utilization,
                            "eval/best_loss": self.best_loss,
                            "eval/best_collision_rate": self.best_collision_rate,
                            "eval/best_codebook_utilization": self.best_codebook_utilization,
                        }
                    )

                # 保存当前检查点，并更新保存队列和堆
                ckpt_path = self._save_checkpoint(
                    epoch_idx,
                    collision_rate=collision_rate,
                    avg_utilization=avg_utilization,
                )
                now_save = (
                    -collision_rate,
                    ckpt_path,
                )  # 以负碰撞率作为优先级，因为 heapq 是最小堆
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)  # 移除最旧的保存
                    self.newest_save_queue.append(now_save)  # 添加最新的保存
                    # 如果当前碰撞率优于堆中保存的最差碰撞率，则替换
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(
                            self.best_save_heap
                        )  # 移除堆中最好的（实际是最差的负数）
                        heapq.heappush(
                            self.best_save_heap, now_save
                        )  # 添加新的保存

                        # 如果被移除的旧保存不在最新保存队列中，则删除对应的文件
                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    # 如果最旧的保存不在最佳保存堆中，则删除对应的文件
                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        if self.use_wandb:
            wandb.finish()  # 结束 WandB 运行

        return self.best_loss, self.best_collision_rate
