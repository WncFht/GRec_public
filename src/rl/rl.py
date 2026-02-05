import argparse
import ast
import atexit
import faulthandler
import os
import signal
import socket
import sys
import threading
import time
import traceback
from collections import defaultdict

from datasets import Dataset as HFDataset

from trl import GRPOConfig

from ..data_rl import FusionSeqRecDataset, SeqRecDataset
from ..parser import parse_dataset_args, parse_global_args, parse_rl_args
from ..utils import ensure_dir, load_model_for_training, set_seed
from .minionerec_trainer import ReReTrainer
from .reward_fns import (
    format_reward,
    initialize_reward_functions,
    ndcg_rule_reward,
    rule_reward,
)

_STAGE = "import"
_HANDLING_SIGNAL = False


def _env_int(name: str, default: int | None = None) -> int | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _rank_info() -> tuple[int | None, int | None, int | None]:
    rank = _env_int("RANK", _env_int("PMI_RANK", _env_int("SLURM_PROCID")))
    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID"))
    world_size = _env_int("WORLD_SIZE", _env_int("PMI_SIZE", _env_int("SLURM_NTASKS")))
    return rank, local_rank, world_size


def _log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    host = os.environ.get("HOSTNAME") or socket.gethostname()
    rank, local_rank, world_size = _rank_info()
    pid = os.getpid()
    prefix = (
        f"[{ts}][{level}][host={host}][pid={pid}]"
        f"[rank={rank} local_rank={local_rank} world_size={world_size}]"
        f"[stage={_STAGE}]"
    )
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _set_stage(stage: str) -> None:
    global _STAGE
    _STAGE = stage
    _log(f"enter stage: {stage}")


def _dump_tracebacks(reason: str) -> None:
    _log(f"Dumping tracebacks ({reason})", level="ERROR")
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        traceback.print_exc()
    sys.stderr.flush()


def _install_debug_hooks() -> None:
    # Ensure we always get Python stack traces on hard failures.
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        # Best-effort; don't break training.
        pass

    def _signal_handler(signum, _frame):
        global _HANDLING_SIGNAL
        if _HANDLING_SIGNAL:
            return
        _HANDLING_SIGNAL = True
        signame = None
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        _log(f"Received signal {signum} ({signame})", level="ERROR")
        _dump_tracebacks(f"signal={signum}")
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    # Optional: `kill -USR1 <pid>` to dump stacks.
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    except Exception:
        pass

    def _excepthook(exc_type, exc, tb):
        _log(f"Uncaught exception: {exc_type.__name__}: {exc}", level="ERROR")
        _dump_tracebacks("sys.excepthook")
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        _log(
            f"Thread exception in {getattr(args, 'thread', None)}: "
            f"{args.exc_type.__name__}: {args.exc_value}",
            level="ERROR",
        )
        _dump_tracebacks("threading.excepthook")

    try:
        threading.excepthook = _thread_excepthook  # type: ignore[attr-defined]
    except Exception:
        pass

    def _atexit():
        _log("Process exiting", level="INFO")

    atexit.register(_atexit)


def _log_env_snapshot() -> None:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "OMP_NUM_THREADS",
        "WANDB_MODE",
        "WANDB_DIR",
        "TRITON_CACHE_DIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    ]
    payload = {k: os.environ.get(k, "<unset>") for k in keys}
    _log(f"env snapshot: {payload}")


def _log_cuda_snapshot() -> None:
    # Best-effort diagnostics; avoid crashing if CUDA isn't usable.
    try:
        import torch  # noqa: PLC0415

        _log(
            "torch cuda snapshot: "
            f"torch={getattr(torch, '__version__', '?')}, "
            f"cuda_available={torch.cuda.is_available()}"
        )
        if torch.cuda.is_available():
            _log(f"torch cuda device_count={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception as e:
                    name = f"<error: {e}>"
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_gb = getattr(props, "total_memory", 0) / 1024**3
                    cc = (
                        f"{getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}"
                    )
                except Exception:
                    total_gb = 0
                    cc = "?"
                _log(f"cuda[{i}]: name={name} cc={cc} total_mem_gb={total_gb:.2f}")
    except Exception as e:
        _log(f"torch cuda snapshot failed: {e}", level="WARNING")


def debug_prefix_index(tokenizer, base_model_name: str):
    r"""
    辅助函数：打印 '### Response:\\nitem\\n' 的分词结果，方便人工选择 prefix_index。
    不会在训练流程中自动调用，如需查看可以在 main 里手动调用。
    """
    sample_item = "<a_1><b_1><c_1><d_1>"
    text = f"### Response:<|im_end|><|im_start|>assistant\n{sample_item}<|im_end|>"
    tokenized = tokenizer(text)
    ids = tokenized["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"[Debug prefix_index] base_model={base_model_name}")
    print("Text:", repr(text))
    print("IDs :", ids)
    print("Tokens:", tokens)


def main():
    _install_debug_hooks()
    _set_stage("parse_args")
    # ====================================================
    # 1. 参数解析 (使用 parser.py)
    # ====================================================
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_rl_args(parser)

    parsed_args = parser.parse_args()  # 扁平对象，传给 utils.* 使用
    num_generations = parsed_args.num_generations
    reward_funcs_cli = parsed_args.reward_funcs
    reward_weights_cli = parsed_args.reward_weights

    print(parsed_args)
    _log_env_snapshot()
    # ====================================================
    # 2. 环境设置 (使用 utils.py)
    # ====================================================
    # # 生成 Run Name
    # run_name = make_run_name(parsed_args)
    # parsed_args.run_name = run_name  # 回写到 args 以供 utils 内部使用

    # 设置 WANDB
    # if parsed_args.run_name and parsed_args.run_name != "none":
    #     os.environ["WANDB_PROJECT"] = "rl_rec"
    #     os.environ["WANDB_RUN_NAME"] = run_name
    # else:
    #     os.environ["WANDB_MODE"] = "disabled"

    # 设置随机种子
    _set_stage("set_seed_and_dirs")
    set_seed(parsed_args.seed)
    ensure_dir(parsed_args.output_dir)

    # print(f"Run Name: {run_name}")
    print(f"Model Type: {parsed_args.model_type}")
    print(f"Base Model: {parsed_args.base_model}")

    # ====================================================
    # 3. 数据集准备
    # ====================================================
    # 先构造 data_rl 里的 PyTorch Dataset，
    # 再统一转换成 Verl 风格记录并包装成 HF Dataset，
    # 以满足 ReReTrainer 的输入格式要求。

    _set_stage("build_datasets")
    tasks = parsed_args.tasks.split(",")
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for task in tasks:
        dataset_list = parsed_args.dataset.split(",")
        for dataset_name in dataset_list:
            train_dataset = None
            valid_dataset = None
            test_dataset = None

            if task.lower() == "seqrec":
                train_dataset = SeqRecDataset(
                    parsed_args,
                    mode="train",
                    dataset=dataset_name,
                )
                if parsed_args.eval_on_valid:
                    valid_dataset = SeqRecDataset(
                        parsed_args,
                        mode="valid",
                        dataset=dataset_name,
                    )
                if parsed_args.eval_on_test:
                    test_dataset = SeqRecDataset(
                        parsed_args,
                        mode="test",
                        dataset=dataset_name,
                    )
            elif task.lower() == "fusionseqrec":
                train_dataset = FusionSeqRecDataset(
                    parsed_args,
                    mode="train",
                    dataset=dataset_name,
                )
                # valid_dataset = FusionSeqRecDataset(
                #     parsed_args,
                #     mode="valid",
                #     dataset=dataset_name,
                # )
                # if parsed_args.eval_on_test:
                #     test_dataset = FusionSeqRecDataset(
                #         parsed_args,
                #         mode="test",
                #         dataset=dataset_name,
                #     )

            if train_dataset is not None:
                train_datasets.append(train_dataset)
                print(
                    f"Task: {task} - dataset: {dataset_name} - train samples: {len(train_dataset)}"
                )
            if test_dataset is not None:
                test_datasets.append(test_dataset)
                print(
                    f"Task: {task} - dataset: {dataset_name} - test samples: {len(test_dataset)}"
                )
            if valid_dataset is not None:
                valid_datasets.append(valid_dataset)
                print(
                    f"Task: {task} - dataset: {dataset_name} - valid samples: {len(valid_dataset)}"
                )

    if not train_datasets:
        msg = "No train datasets constructed. Please check `--tasks` and `--dataset`."
        raise ValueError(msg)

    # ====================================================
    # 4. 模型加载 (使用 utils.load_model_for_training)
    # ====================================================
    # 这个函数封装了: Tokenizer, Resize Embeddings, LoRA, Freeze

    _set_stage("load_model_for_training")
    model, processor, orig_vocab, new_vocab, new_tokens, embedding_hooks = (
        load_model_for_training(
            args=parsed_args,
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        )
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
    print(f"Using eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"Using pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    if parsed_args.debug_prefix_index:
        debug_prefix_index(tokenizer, parsed_args.base_model)
        return

    if new_tokens and len(new_tokens) > 0:
        _log(
            f"Tokenizer extended with {len(new_tokens)} new tokens. "
            "If you are resuming from an SFT checkpoint, this often means the current "
            "`--index_file/--dataset` token space differs from what the checkpoint was trained on, "
            "and rule/ndcg rewards may become ~0 due to exact-match evaluation."
        )

    # 注册 tokenizer 并初始化奖励函数所需的上下文
    if initialize_reward_functions(
        num_generations,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
    ):
        return

    # ====================================================
    # 3.1 转换数据集为 Verl 记录（包含 ground_truth token ids）
    # ====================================================
    _set_stage("build_train_records")
    print("Processing Train Dataset (to Verl records)...")
    train_records = []
    for ds in train_datasets:
        if hasattr(ds, "to_verl_records"):
            train_records.extend(ds.to_verl_records("train", tokenizer=tokenizer))

    train_dataset = HFDataset.from_list(train_records)
    train_dataset = train_dataset.shuffle(seed=parsed_args.seed)

    test_eval_dataset = None
    if parsed_args.eval_on_test and test_datasets:
        _set_stage("build_test_records")
        print("Processing Test Dataset (to Verl records)...")
        test_records = []
        for ds in test_datasets:
            if hasattr(ds, "to_verl_records"):
                test_records.extend(ds.to_verl_records("test", tokenizer=tokenizer))
        test_eval_dataset = HFDataset.from_list(test_records) if test_records else None

    valid_eval_dataset = None
    if parsed_args.eval_on_valid and valid_datasets:
        _set_stage("build_valid_records")
        print("Processing Valid Dataset (to Verl records)...")
        valid_records = []
        for ds in valid_datasets:
            if hasattr(ds, "to_verl_records"):
                valid_records.extend(ds.to_verl_records("valid", tokenizer=tokenizer))
        valid_eval_dataset = (
            HFDataset.from_list(valid_records) if valid_records else None
        )

    combined_eval_dataset = test_eval_dataset
    if parsed_args.eval_on_valid:
        if valid_eval_dataset is not None and test_eval_dataset is not None:
            combined_eval_dataset = {
                "test": test_eval_dataset,
                "valid": valid_eval_dataset,
            }
        elif valid_eval_dataset is not None:
            combined_eval_dataset = valid_eval_dataset

    print(f"Train Size: {len(train_dataset)}")
    print(
        f"Test Eval Size: {len(test_eval_dataset) if test_eval_dataset is not None else 0}"
    )
    if parsed_args.eval_on_valid:
        print(
            f"Valid Eval Size: {len(valid_eval_dataset) if valid_eval_dataset is not None else 0}"
        )

    # if True:
    #     debug_prefix_index(tokenizer, "test")
    #     sys.exit()
    # ====================================================
    # 4.1 基于数据集构建 hash_dict（前缀约束）
    # ====================================================
    # 简单的 prefix_index 规则（与原实现保持一致），
    # 如需更精细可以用下方 debug 函数做检查后手动调整。
    _set_stage("build_hash_dict")
    base_model_lower = parsed_args.base_model.lower()
    if "llava" in base_model_lower:
        prefix_index = 7
    elif "gpt2" in base_model_lower:
        prefix_index = 4
    else:
        prefix_index = 3

    merged_hash_dict: dict[str, set[int]] = defaultdict(set)
    for ds in train_datasets:
        if hasattr(ds, "build_hash_dict"):
            ds_hash = ds.build_hash_dict(tokenizer, prefix_index=prefix_index)
            for k, vals in ds_hash.items():
                merged_hash_dict[k].update(vals)

    hash_dict = {k: sorted(list(v)) for k, v in merged_hash_dict.items()}
    print(f"Built hash_dict entries: {len(hash_dict)} with prefix_index={prefix_index}")

    # print("10th of the hash_dict")
    # import pprint; pprint.pprint(dict(list(hash_dict.items())[:10]))
    def _parse_list_arg(val, cast=None):
        if not val:
            return []
        if isinstance(val, (list, tuple)):
            items = list(val)
        else:
            try:
                lit = ast.literal_eval(val)
                if isinstance(lit, (list, tuple)):
                    items = list(lit)
                else:
                    items = [lit]
            except Exception:
                items = [v for v in str(val).split(",") if v != ""]
        if cast:
            items = [cast(x) for x in items]
        return items

    def _build_reward_registry(fmt_pattern: str | None):
        registry = {
            "format": format_reward,
            "rule": rule_reward,
            "ndcg": ndcg_rule_reward,
        }
        return registry

    reward_fun: list = []
    reward_weights_list: list[float] | None = None
    parsed_funcs = _parse_list_arg(reward_funcs_cli)
    parsed_weights = _parse_list_arg(reward_weights_cli, cast=float)

    if parsed_funcs:
        registry = _build_reward_registry(None)
        for name in parsed_funcs:
            if name not in registry:
                raise ValueError(
                    f"Unknown reward_func '{name}'. 可选: {list(registry.keys())}"
                )
            reward_fun.append(registry[name])
        print("Using reward_funcs from CLI:", parsed_funcs)
        if parsed_weights:
            if len(parsed_weights) != len(reward_fun):
                raise ValueError(
                    f"reward_weights 长度 {len(parsed_weights)} 必须与 reward_funcs {len(reward_fun)} 相同"
                )
            reward_weights_list = parsed_weights
            print("Using reward_weights from CLI:", reward_weights_list)
    else:
        print("Failed to parse reward_funcs from CLI, using --reward_type instead.")
        reward_type = parsed_args.reward_type
        if reward_type == "rule":
            reward_fun = [format_reward, rule_reward]
        elif reward_type == "ranking":
            reward_fun = [format_reward, rule_reward, ndcg_rule_reward]
        elif reward_type == "ranking_only":
            reward_fun = [format_reward, ndcg_rule_reward]

    # ====================================================
    # 6. 配置 Trainer
    # ====================================================
    # 映射参数到 GRPOConfig
    _set_stage("build_training_args")
    training_args = GRPOConfig(
        output_dir=parsed_args.output_dir,
        save_steps=0.1,
        save_total_limit=20,
        save_only_model=True,
        eval_strategy="steps",
        max_completion_length=parsed_args.max_completion_length,
        num_generations=num_generations,
        temperature=parsed_args.temperature,
        sync_ref_model=parsed_args.sync_ref_model,
        per_device_eval_batch_size=parsed_args.eval_batch_size,
        per_device_train_batch_size=parsed_args.train_batch_size,
        gradient_accumulation_steps=parsed_args.gradient_accumulation_steps,
        eval_steps=parsed_args.eval_step,
        logging_steps=5,
        log_completions=parsed_args.log_completions,
        learning_rate=parsed_args.learning_rate,
        beta=parsed_args.beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        num_train_epochs=parsed_args.num_train_epochs,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to="wandb",
        eval_on_start=True,
    )
    training_args.completion_log_interval = parsed_args.completion_log_interval
    training_args.clip = parsed_args.clip
    training_args.clip_ratio = parsed_args.clip_ratio
    training_args.clip_ratio_low = parsed_args.clip_ratio_low
    training_args.clip_ratio_high = parsed_args.clip_ratio_high
    training_args.clip_ratio_c = parsed_args.clip_ratio_c
    training_args.reward_weights = reward_weights_list

    # 初始化自定义 Trainer
    _set_stage("init_trainer")
    _log_cuda_snapshot()
    t0 = time.perf_counter()
    try:
        trainer = ReReTrainer(
            model=model,
            base_model=parsed_args.base_model,
            dapo=parsed_args.dapo,
            gspo=parsed_args.gspo,
            noscale=parsed_args.noscale,
            nodemean=parsed_args.nodemean,
            use_prm=parsed_args.use_prm,
            prm_match_mode=parsed_args.prm_match_mode,
            use_sft_loss=parsed_args.use_sft_loss,
            sft_loss_coef=parsed_args.sft_loss_coef,
            add_gt=parsed_args.add_gt,
            dynamic_sampling=parsed_args.dynamic_sampling,
            beam_search=parsed_args.beam_search,
            test_during_training=parsed_args.test_during_training,
            test_beam=parsed_args.test_beam,
            hash_dict=hash_dict,
            prefix_index=prefix_index,
            reward_funcs=reward_fun,
            train_dataset=train_dataset,
            eval_dataset=combined_eval_dataset,
            processing_class=tokenizer,
            args=training_args,
        )
    except Exception as e:
        _log(f"ReReTrainer init failed: {e}", level="ERROR")
        _dump_tracebacks("ReReTrainer init exception")
        raise
    finally:
        _log(f"ReReTrainer init elapsed_s={time.perf_counter() - t0:.2f}")

    # ====================================================
    # 7. 训练与保存
    # ====================================================
    _set_stage("train_begin")
    print("Starting Training...")
    t1 = time.perf_counter()
    try:
        trainer.train()
    except Exception as e:
        _log(f"trainer.train() failed: {e}", level="ERROR")
        _dump_tracebacks("trainer.train exception")
        raise
    finally:
        _log(f"trainer.train elapsed_s={time.perf_counter() - t1:.2f}")

    _set_stage("save_and_eval")
    print(f"Saving model to {parsed_args.output_dir}")
    trainer.save_model(parsed_args.output_dir)

    if test_eval_dataset is not None:
        print("Running evaluation on test split...")
        test_metrics = trainer.evaluate(
            eval_dataset=test_eval_dataset, metric_key_prefix="test"
        )
        print(f"Test metrics: {test_metrics}")
    if parsed_args.eval_on_valid and valid_eval_dataset is not None:
        print("Running evaluation on valid split...")
        valid_metrics = trainer.evaluate(
            eval_dataset=valid_eval_dataset, metric_key_prefix="valid"
        )
        print(f"Valid metrics: {valid_metrics}")

    # 保存最终 checkpoint
    final_dir = os.path.join(parsed_args.output_dir, "final_checkpoint")
    ensure_dir(final_dir)

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 保存 token metadata
    if hasattr(model, "config"):
        model.config.save_pretrained(final_dir)

    print("Training Finished.")


if __name__ == "__main__":
    main()
