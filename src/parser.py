import argparse


def parse_global_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    global_args = parser.add_argument_group("global_args")
    global_args.add_argument("--seed", type=int, default=42, help="Random seed")
    global_args.add_argument(
        "--model_type",
        type=str,
        default="qwen2_vl",
        required=True,
        choices=[
            "qwen2_vl",
            "qwen2_5_vl",
            "llava_onevision",
            "qwen2",
            "qwen2_5",
            "qwen",
            "llama",
        ],
        help="模型类型 (qwen2_vl or qwen2_5_vl, llava_onevision)",
    )

    return parser


def parse_rl_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    rl_parser = parser.add_argument_group("rl_args")

    rl_parser.add_argument(
        "--base_model", type=str, default="", help="Base Model."
    )
    rl_parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for RL checkpoints.",
    )
    rl_parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Per-device train batch size for RL.",
    )
    rl_parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Per-device eval batch size for RL.",
    )
    rl_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    rl_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature during RL decoding.",
    )
    rl_parser.add_argument(
        "--max_completion_length",
        type=int,
        default=128,
        help="Maximum number of tokens to generate for each completion during RL.",
    )
    rl_parser.add_argument(
        "--add_gt",
        action="store_true",
        default=False,
        help="Whether to add ground-truth completion into candidates.",
    )
    rl_parser.add_argument(
        "--eval_step",
        type=float,
        default=0.199,
        help="Evaluation frequency (in epochs or ratio).",
    )
    rl_parser.add_argument(
        "--log_completions",
        action="store_true",
        default=False,
        help="Whether to log rollout prompts/completions and rewards to wandb.",
    )
    rl_parser.add_argument(
        "--completion_log_interval",
        type=int,
        default=50,
        help="Step interval for logging rollouts (independent from --logging_steps).",
    )
    rl_parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="Number of generations (e.g. beams) per prompt.",
    )
    rl_parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs for RL.",
    )
    rl_parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for RL optimizer.",
    )
    rl_parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Beta coefficient in GRPO loss.",
    )
    rl_parser.add_argument(
        "--beam_search",
        action="store_true",
        default=False,
        help="Whether to use beam search during RL.",
    )
    rl_parser.add_argument(
        "--test_during_training",
        action="store_true",
        default=True,
        help="Whether to run test/eval during training.",
    )
    rl_parser.add_argument(
        "--eval_on_test",
        action="store_true",
        default=False,
        help="训练结束后是否额外在 test split 上跑一轮评估。",
    )
    rl_parser.add_argument(
        "--dynamic_sampling",
        action="store_true",
        default=False,
        help="Enable dynamic sampling strategy for RL data.",
    )
    rl_parser.add_argument(
        "--mask_all_zero",
        action="store_true",
        default=False,
        help="Whether to mask all-zero rewards.",
    )
    rl_parser.add_argument(
        "--sync_ref_model",
        action="store_true",
        default=False,
        help="Whether to periodically sync reference model.",
    )
    rl_parser.add_argument(
        "--test_beam",
        type=int,
        default=20,
        help="Beam size used at test time.",
    )
    rl_parser.add_argument(
        "--reward_type",
        type=str,
        default="rule",
        help="Reward type: rule|ranking|ranking_only|semantic|sasrec.",
    )
    rl_parser.add_argument(
        "--sample_train",
        action="store_true",
        default=False,
        help="Whether to subsample training data.",
    )
    rl_parser.add_argument(
        "--dapo",
        action="store_true",
        default=False,
        help="Enable DAPO training variant.",
    )
    rl_parser.add_argument(
        "--gspo",
        action="store_true",
        default=False,
        help="Enable GSPO training variant.",
    )
    rl_parser.add_argument(
        "--use_sft_loss",
        action="store_true",
        default=False,
        help="Mix in an auxiliary SFT loss during RL updates.",
    )
    rl_parser.add_argument(
        "--sft_loss_coef",
        type=float,
        default=1e-3,
        help="Weight for the auxiliary SFT loss when enabled.",
    )
    rl_parser.add_argument(
        "--debug_prefix_index",
        action="store_true",
        default=False,
        help="Print tokenization details for '### Response:' to help choose prefix_index.",
    )
    rl_parser.add_argument(
        "--train_prompt_sample_num",
        type=str,
        required=True,
        default="1,1,1,1,1,1",
        help="the number of sampling prompts for each task",
    )
    rl_parser.add_argument(
        "--train_data_sample_num",
        type=str,
        required=True,
        default="0,0,0,100000,0,0",
        help="the number of sampling prompts for each task",
    )
    rl_parser.add_argument("--fp16", action="store_true", default=False)
    rl_parser.add_argument("--bf16", action="store_true", default=False)

    rl_parser.add_argument(
        "--noscale",
        action="store_true",
        default=False,
        help="If set, do not divide advantages by std when normalizing rewards.",
    )

    return parser


def parse_dataset_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    dataset_args = parser.add_argument_group("dataset_args")
    dataset_args.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="data directory",
    )
    dataset_args.add_argument(
        "--image_path",
        type=str,
        default="images",
        help="image directory",
    )
    dataset_args.add_argument(
        "--tasks",
        type=str,
        default="seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain,mmitem2index,mmindex2item,mmitemenrich",
        help="Downstream tasks, separate by comma",
    )
    dataset_args.add_argument(
        "--dataset",
        type=str,
        default="Instruments",
        help="Dataset name",
    )
    dataset_args.add_argument(
        "--index_file",
        type=str,
        required=True,
        default=".index_qwen7B.json",
        help="the item indices file",
    )

    # arguments related to sequential task
    dataset_args.add_argument(
        "--max_his_len",
        type=int,
        default=20,
        help="the max number of items in history sequence, -1 means no limit",
    )
    dataset_args.add_argument(
        "--add_prefix",
        action="store_true",
        default=False,
        help="whether add sequential prefix in history",
    )
    dataset_args.add_argument(
        "--his_sep",
        type=str,
        default=", ",
        help="The separator used for history",
    )

    dataset_args.add_argument(
        "--valid_prompt_id",
        type=int,
        default=0,
        help="The prompt used for validation",
    )
    dataset_args.add_argument(
        "--sample_valid",
        action="store_true",
        default=True,
        help="use sampled prompt for validation",
    )
    dataset_args.add_argument(
        "--valid_prompt_sample_num",
        type=int,
        default=2,
        help="the number of sampling validation sequential recommendation prompts",
    )
    dataset_args.add_argument(
        "--ratio_dataset",
        type=float,
        default=1.0,
        help="the ratio of dataset",
    )

    dataset_args.add_argument(
        "--only_train_response",
        action="store_true",
        default=True,
        help="whether only train on responses",
    )

    return parser


def parse_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    train_args = parser.add_argument_group("train_args")
    train_args.add_argument(
        "--base_model",
        type=str,
        required=True,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="basic model path",
    )
    train_args.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="./ckpt/",
        help="The output directory",
    )
    train_args.add_argument(
        "--freeze",
        type=str,
        default=None,
        choices=["all", "visual", "embeddings", "only_embeddings"],
    )

    train_args.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="The name of the optimizer",
    )
    train_args.add_argument("--epochs", type=int, required=True, default=4)
    train_args.add_argument("--learning_rate", type=float, default=2e-5)
    train_args.add_argument(
        "--per_device_batch_size",
        type=int,
        required=True,
        default=8,
    )
    train_args.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=False,
    )
    train_args.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    train_args.add_argument("--logging_step", type=int, default=1)
    train_args.add_argument("--model_max_length", type=int, default=2048)
    train_args.add_argument("--weight_decay", type=float, default=0.01)
    train_args.add_argument("--run_name", type=str, default=None)

    # lora 的先不管了
    train_args.add_argument("--use_lora", action="store_true", default=False)
    train_args.add_argument("--lora_r", type=int, default=32)
    train_args.add_argument("--lora_alpha", type=int, default=32)
    train_args.add_argument("--lora_dropout", type=float, default=0.05)
    train_args.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        help="separate by comma",
    )
    train_args.add_argument(
        "--lora_modules_to_save",
        type=str,
        default="embed_tokens,lm_head",
        help="separate by comma",
    )

    train_args.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="either training checkpoint or final adapter",
    )

    train_args.add_argument("--warmup_ratio", type=float, default=0.01)
    train_args.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train_args.add_argument(
        "--save_and_eval_strategy", type=str, default="epoch"
    )
    train_args.add_argument("--save_and_eval_steps", type=int, default=1000)
    train_args.add_argument("--fp16", action="store_true", default=False)
    train_args.add_argument("--bf16", action="store_true", default=False)
    train_args.add_argument(
        "--deepspeed", type=str, default="./config/ds_z3_bf16.json"
    )

    train_args.add_argument(
        "--train_prompt_sample_num",
        type=str,
        required=True,
        default="1,1,1,1,1,1",
        help="the number of sampling prompts for each task",
    )
    train_args.add_argument(
        "--train_data_sample_num",
        type=str,
        required=True,
        default="0,0,0,100000,0,0",
        help="the number of sampling prompts for each task",
    )
    train_args.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers for data loading",
    )

    return parser


def parse_test_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    test_args = parser.add_argument_group("test_args")
    test_args.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    test_args.add_argument(
        "--model_name",
        type=str,
        default="default_model",
    )
    test_args.add_argument(
        "--base_model",
        type=str,
        default="./ckpt/base_model/Qwen2-VL-2B-Instruct",
        help="基础模型路径（仅在使用LoRA时需要）",
    )
    test_args.add_argument("--lora", action="store_true", default=False)
    test_args.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="The checkpoint path",
    )
    test_args.add_argument(
        "--filter_items",
        action="store_true",
        default=False,
        help="whether filter illegal items",
    )

    test_args.add_argument(
        "--results_file",
        type=str,
        default="./results/results.json",
        help="result output path",
    )

    test_args.add_argument("--test_batch_size", type=int, default=1)
    test_args.add_argument("--num_beams", type=int, default=20)
    test_args.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help="test sample number, -1 represents using all test data",
    )
    test_args.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID when testing with single GPU",
    )
    test_args.add_argument(
        "--test_prompt_ids",
        type=str,
        default="0",
        help="test prompt ids, separate by comma. 'all' represents using all",
    )
    test_args.add_argument(
        "--metrics",
        type=str,
        default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        help="test metrics, separate by comma",
    )
    test_args.add_argument("--test_task", type=str, default="SeqRec")
    test_args.add_argument(
        "--benchmark_metrics",
        type=str,
        default="bleu,rouge",
        help="test metrics for text generation benchmark, separate by comma",
    )
    test_args.add_argument(
        "--use_constrained_generation",
        action="store_true",
        default=False,
    )
    test_args.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
    )
    test_args.add_argument(
        "--print_freq",
        type=int,
        default=4,
    )
    test_args.add_argument(
        "--reference_data_path",
        type=str,
        default="./data/Instruments/Instruments.item_enriched_v2.json",
        help="参考数据(ground truth)的路径",
    )

    return parser
