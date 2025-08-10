import datetime
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from data import (
    FusionSeqRecDataset,
    ItemFeatDataset,
    ItemSearchDataset,
    MultimodalDataset,
    PreferenceObtainDataset,
    SeqRecDataset,
    TextEnrichDataset,
)


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen_vl",
        choices=["qwen_vl", "t5"],
        help="模型类型 (qwen_vl or t5)",
    )

    parser.add_argument(
        "--base_model", type=str, default="./llama-7b/", help="basic model path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./ckpt/", help="The output directory"
    )

    return parser


def parse_dataset_args(parser):
    parser.add_argument(
        "--data_path", type=str, default="", help="data directory"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain,mmitem2index,mmindex2item,mmitemenrich",
        help="Downstream tasks, separate by comma",
    )
    parser.add_argument(
        "--dataset", type=str, default="Games", help="Dataset name"
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=".index.json",
        help="the item indices file",
    )

    # arguments related to sequential task
    parser.add_argument(
        "--max_his_len",
        type=int,
        default=20,
        help="the max number of items in history sequence, -1 means no limit",
    )
    parser.add_argument(
        "--add_prefix",
        action="store_true",
        default=False,
        help="whether add sequential prefix in history",
    )
    parser.add_argument(
        "--his_sep",
        type=str,
        default=", ",
        help="The separator used for history",
    )
    parser.add_argument(
        "--only_train_response",
        action="store_true",
        default=False,
        help="whether only train on responses",
    )

    parser.add_argument(
        "--train_prompt_sample_num",
        type=str,
        default="1,1,1,1,1,1",
        help="the number of sampling prompts for each task",
    )
    parser.add_argument(
        "--train_data_sample_num",
        type=str,
        default="0,0,0,100000,0,0",
        help="the number of sampling prompts for each task",
    )

    parser.add_argument(
        "--valid_prompt_id",
        type=int,
        default=0,
        help="The prompt used for validation",
    )
    parser.add_argument(
        "--sample_valid",
        action="store_true",
        default=True,
        help="use sampled prompt for validation",
    )
    parser.add_argument(
        "--valid_prompt_sample_num",
        type=int,
        default=2,
        help="the number of sampling validation sequential recommendation prompts",
    )

    return parser


def parse_train_args(parser):
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="The name of the optimizer",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        help="separate by comma",
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default="embed_tokens,lm_head",
        help="separate by comma",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="either training checkpoint or final adapter",
    )

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed", type=str, default="./config/ds_z3_bf16.json"
    )

    return parser


def parse_test_args(parser):
    parser.add_argument(
        "--ckpt_path", type=str, default="", help="The checkpoint path"
    )
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument(
        "--filter_items",
        action="store_true",
        default=False,
        help="whether filter illegal items",
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default="./results/test-ddp.json",
        help="result output path",
    )

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help="test sample number, -1 represents using all test data",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID when testing with single GPU",
    )
    parser.add_argument(
        "--test_prompt_ids",
        type=str,
        default="0",
        help="test prompt ids, separate by comma. 'all' represents using all",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        help="test metrics, separate by comma",
    )
    parser.add_argument("--test_task", type=str, default="SeqRec")

    return parser


# def parse_test_args(parser):
#     """解析测试相关参数"""
#     parser.add_argument('--ckpt_path', type=str, required=True,
#                     help='Model checkpoint path')
#     parser.add_argument('--base_model', type=str, required=True,
#                     help='Base model path (for LoRA)')
#     parser.add_argument('--lora', action='store_true',
#                     help='Whether to use LoRA model')
#     parser.add_argument('--gpu_id', type=int, default=0,
#                     help='GPU ID')
#     parser.add_argument('--test_task', type=str, default='seqrec',
#                     choices=['seqrec', 'itemsearch', 'fusionseqrec', 'mmitem2index', 'mmindex2item'],
#                     help='Test task type')
#     parser.add_argument('--test_prompt_ids', type=str, default='all',
#                     help='Test prompt IDs, comma-separated or "all"')
#     parser.add_argument('--test_batch_size', type=int, default=1,
#                     help='Test batch size')
#     parser.add_argument('--num_beams', type=int, default=5,
#                     help='Number of beams for beam search')
#     parser.add_argument('--max_new_tokens', type=int, default=10,
#                     help='Maximum number of new tokens to generate')
#     parser.add_argument('--metrics', type=str, default='hit@1,hit@5,hit@10,ndcg@5,ndcg@10',
#                     help='Evaluation metrics, comma-separated')
#     parser.add_argument('--results_file', type=str, default='results/test_results.json',
#                     help='Results file save path')
#     parser.add_argument('--filter_items', action='store_true',
#                     help='Whether to filter items not in candidate set')
#     parser.add_argument('--print_freq', type=int, default=10,
#                     help='Print frequency')

#     return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):
    tasks = args.tasks.split(",")

    train_prompt_sample_num = [
        int(_) for _ in args.train_prompt_sample_num.split(",")
    ]  # 1,1,1,1
    assert len(tasks) == len(
        train_prompt_sample_num
    ), "prompt sample number does not match task number"
    train_data_sample_num = [
        int(_) for _ in args.train_data_sample_num.split(",")
    ]
    assert len(tasks) == len(
        train_data_sample_num
    ), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(
        tasks, train_prompt_sample_num, train_data_sample_num, strict=False
    ):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "item2index" or task.lower() == "index2item":
            dataset = ItemFeatDataset(
                args,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "fusionseqrec":
            dataset = FusionSeqRecDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "itemsearch":
            dataset = ItemSearchDataset(
                args,
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "preferenceobtain":
            dataset = PreferenceObtainDataset(
                args,
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "mmitem2index" or task.lower() == "mmindex2item":
            dataset = MultimodalDataset(
                args,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "mmitemenrich":
            dataset = TextEnrichDataset(
                args,
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args, "valid", args.valid_prompt_sample_num)

    return train_data, valid_data


def load_test_dataset(args):
    if args.test_task.lower() == "seqrec":
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
        # test_data = SeqRecTestDataset(args, sample_num=args.sample_num)
    elif args.test_task.lower() == "itemsearch":
        test_data = ItemSearchDataset(
            args, mode="test", sample_num=args.sample_num
        )
    elif args.test_task.lower() == "fusionseqrec":
        test_data = FusionSeqRecDataset(
            args, mode="test", sample_num=args.sample_num
        )
    else:
        raise NotImplementedError

    return test_data


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def verify_token_ordering(processor, original_vocab_size, new_tokens):
    """验证新添加的token是否真的在词汇表末尾"""
    print("=== 验证Token排序 ===")

    # 检查原始token的一些示例
    original_samples = [0, 1, 100, 1000, original_vocab_size - 1]
    print("原始token示例:")
    for token_id in original_samples:
        if token_id < original_vocab_size:
            token = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"  ID {token_id}: '{token}'")

    # 检查新添加的token
    print(f"\n新添加的token (总共{len(new_tokens)}个):")
    current_vocab_size = len(processor.tokenizer)
    new_token_start = original_vocab_size

    # 验证前几个新token
    for i, expected_token in enumerate(new_tokens[:5]):  # 只显示前5个
        token_id = new_token_start + i
        if token_id < current_vocab_size:
            actual_token = processor.tokenizer.convert_ids_to_tokens(
                [token_id]
            )[0]
            print(
                f"  ID {token_id}: 期望 '{expected_token}' -> 实际 '{actual_token}' {'✓' if expected_token == actual_token else '✗'}"
            )

    # 验证最后几个新token
    if len(new_tokens) > 5:
        print("  ...")
        for i in range(max(0, len(new_tokens) - 3), len(new_tokens)):
            expected_token = new_tokens[i]
            token_id = new_token_start + i
            if token_id < current_vocab_size:
                actual_token = processor.tokenizer.convert_ids_to_tokens(
                    [token_id]
                )[0]
                print(
                    f"  ID {token_id}: 期望 '{expected_token}' -> 实际 '{actual_token}' {'✓' if expected_token == actual_token else '✗'}"
                )

    # 最终验证
    print("\n验证结果:")
    print(f"  原始词汇表大小: {original_vocab_size}")
    print(f"  当前词汇表大小: {current_vocab_size}")
    print(f"  新增token数量: {len(new_tokens)}")
    print(
        f"  预期新token ID范围: {original_vocab_size} ~ {current_vocab_size - 1}"
    )

    return (
        new_token_start == original_vocab_size
        and current_vocab_size == original_vocab_size + len(new_tokens)
    )


def freeze_original_embeddings_simple(model, original_vocab_size):
    """简单版本：直接冻结原有embedding参数"""
    input_embeddings = model.get_input_embeddings()

    # 冻结原有token的embedding
    with torch.no_grad():
        # 将原有embedding参数的requires_grad设为False
        original_embeddings = input_embeddings.weight[:original_vocab_size]
        original_embeddings.requires_grad_(False)

        # 确保新增embedding参数可以训练
        new_embeddings = input_embeddings.weight[original_vocab_size:]
        new_embeddings.requires_grad_(True)

    print(f"冻结了前 {original_vocab_size} 个token的embedding参数")
    print(
        f"保持后 {len(input_embeddings.weight) - original_vocab_size} 个新token的embedding可训练"
    )


def freeze_original_embeddings_for_lora(model, original_vocab_size):
    """适用于LoRA+modules_to_save的选择性冻结"""
    # 找到modules_to_save中的embed_tokens
    for name, module in model.named_modules():
        if "embed_tokens" in name and hasattr(module, "weight"):
            print(f"找到embedding层: {name}, shape: {module.weight.shape}")

            # 冻结原始token (0 到 original_vocab_size-1)
            module.weight[:original_vocab_size].requires_grad_(False)

            # 保持新token可训练 (original_vocab_size 到 end)
            module.weight[original_vocab_size:].requires_grad_(True)

            print(f"冻结了前 {original_vocab_size} 个token")
            print(
                f"保持后 {module.weight.shape[0] - original_vocab_size} 个新token可训练"
            )
            break


def freeze_original_embeddings_with_hook(model, original_vocab_size):
    """
    使用梯度hook冻结原始embedding参数
    """
    hooks = []

    def create_embedding_hook(vocab_size):
        def hook_fn(grad):
            if grad is not None:
                # 创建新的梯度，原始token位置置零
                new_grad = grad.clone()
                new_grad[:vocab_size] = 0.0
                return new_grad
            return grad

        return hook_fn

    # 为embedding注册hook
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "embed_tokens" in name:
                handle = param.register_hook(
                    create_embedding_hook(original_vocab_size)
                )
                hooks.append(handle)
                print(f"为 {name} 注册embedding hook, shape: {param.shape}")

    print(
        f"注册了 {len(hooks)} 个梯度hook来冻结前 {original_vocab_size} 个token"
    )
    return hooks
