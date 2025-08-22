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
        choices=["qwen2_vl", "qwen2_5_vl", "llava_onevision"],
        help="模型类型 (qwen2_vl or qwen2_5_vl, llava_onevision)",
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
        required=True,
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
        choices=["all", "visual", "embeddings"],
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
        "--gradient_accumulation_steps", type=int, default=2
    )
    train_args.add_argument("--logging_step", type=int, default=10)
    train_args.add_argument("--model_max_length", type=int, default=2048)
    train_args.add_argument("--weight_decay", type=float, default=0.01)

    # lora 的先不管了
    train_args.add_argument("--use_lora", action="store_true", default=False)
    train_args.add_argument("--lora_r", type=int, default=8)
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

    return parser


def parse_test_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    test_args = parser.add_argument_group("test_args")
    test_args.add_argument(
        "--ckpt_path", type=str, default="", help="The checkpoint path"
    )
    test_args.add_argument("--lora", action="store_true", default=False)
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

    return parser
