from dataclasses import dataclass, field


@dataclass
class TrainingSample:
    """
    数据集返回的单个样本的统一结构。
    所有 Dataset 的 __getitem__ 都应返回此结构。
    这个结构是扁平的，直接包含模型处理所需的原始信息。
    Collator 会负责将这些信息转换成模型所需的张量。
    """

    # 核心字段
    input_text: str
    label_text: str

    # 多模态相关
    is_multimodal: bool = False
    image_path: str | list[str] | None = None

    # 可选元数据
    # 用于评估或调试，例如在 evaluate.py 中按item_id查找参考答案
    item_id: str | None = None


@dataclass
class EnrichedData:
    """
    用于文本丰富任务的完整数据结构，
    既作为 TextEnrichDataset 的内部存储，也作为评估时的参考数据。
    """

    item_id: str
    item: str
    title: str
    description: str
    brand: str
    categories: str
    enhanced_title: str
    tags: str
    highlights: str
    characteristics: str
    image_path: str


@dataclass
class GlobalArgs:
    """
    全局配置，影响整个项目的基本行为

    Attributes
    ----------
        seed (int): 全局随机种子, 用于保证实验可复现.
        model_type (str): 模型类型, 可选 "qwen_vl", "t5", "llama".
        base_model (str): 基础大模型的本地路径.
        output_dir (str): 训练检查点和输出文件的保存目录.

    """

    seed: int = 42
    model_type: str = "qwen_vl"
    base_model: str = "./llama-7b/"
    output_dir: str = "./ckpt/"
    debug: bool = False


@dataclass
class BenchmarkModelArgs:
    """
    用于 benchmark 的单个模型的配置
    """

    name: str
    enabled: bool = True
    model_type: str = "qwen_vl"
    path: str = ""
    lora: bool = False
    ckpt_path: str = ""
    # 可选，用于覆盖 dataset_args 中的全局 index_file
    index_file: str | None = None


@dataclass
class DatasetArgs:
    """
    数据集相关的配置

    Attributes
    ----------
        data_path (str): 数据集文件所在的根目录.
        image_path (str): 数据集根目录下存放图像的子目录名称.
        tasks (str): 需要执行的任务列表, 用逗号分隔.
        dataset (str): 使用的数据集名称.
        index_file (str): 存储物品索引的映射文件名.
        max_his_len (int): 用户历史序列的最大长度, -1表示不限制.
        add_prefix (bool): 是否在历史序列前添加 "1:, 2:, ..." 这样的前缀.
        his_sep (str): 历史序列中物品之间的分隔符.
        only_train_response (bool): 训练时是否只计算响应部分(答案)的loss.
        train_prompt_sample_num (str): 训练时为每个任务采样多少种prompt, 与tasks对应.
        train_data_sample_num (str): 训练时为每个任务采样多少条数据, 0表示全部使用, 与tasks对应.
        valid_prompt_id (int): 验证时使用的prompt的ID.
        sample_valid (bool): 验证时是否从多种prompt中采样.
        valid_prompt_sample_num (int): 验证时采样的prompt数量(如果sample_valid为True).

    """

    data_path: str = ""
    image_path: str = "images"
    tasks: str = "seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain,mmitem2index,mmindex2item,mmitemenrich,mmitemenrichwithoutid"
    dataset: str = "Instruments"
    index_file: str = ".index_qwen7B.json"
    max_his_len: int = 20
    add_prefix: bool = False
    his_sep: str = ", "
    only_train_response: bool = False
    train_prompt_sample_num: str = "1,1,1,1,1,1"
    train_data_sample_num: str = "0,0,0,100000,0,0"
    valid_prompt_id: int = 0
    sample_valid: bool = True
    valid_prompt_sample_num: int = 2
    ratio_dataset: float = 1.0


@dataclass
class TrainingArgs:
    """
    训练过程相关的配置

    Attributes
    ----------
        optim (str): 使用的优化器名称.
        epochs (int): 训练的总轮数.
        learning_rate (float): 学习率.
        per_device_batch_size (int): 每个GPU设备上的批处理大小.
        gradient_accumulation_steps (int): 梯度累积步数, 用于模拟更大的batch size.
        logging_step (int): 每隔多少步记录一次日志.
        model_max_length (int): 模型能处理的最大序列长度.
        weight_decay (float): 权重衰减系数.
        use_lora (bool): 是否使用LoRA.
        lora_r (int): LoRA的秩.
        lora_alpha (int): LoRA的alpha参数.
        lora_dropout (float): LoRA层的dropout概率.
        lora_target_modules (str): LoRA作用的目标模块, 用逗号分隔.
        lora_modules_to_save (str): 除了LoRA层外, 额外需要训练和保存的模块, 用逗号分隔.
        resume_from_checkpoint (Optional[str]): 从指定的检查点继续训练.
        warmup_ratio (float): 学习率预热阶段占总训练步数的比例.
        lr_scheduler_type (str): 学习率调度器类型.
        save_and_eval_strategy (str): 模型保存和评估策略, "epoch"或"steps".
        save_and_eval_steps (int): 如果策略是"steps", 则每隔N步保存一次.
        fp16 (bool): 是否启用FP16混合精度训练.
        bf16 (bool): 是否启用BF16混合精度训练.
        deepspeed (str): DeepSpeed配置文件的路径.
        device (str): 训练设备, "cuda"或"cpu".

    """

    optim: str = "adamw_torch"
    epochs: int = 4
    learning_rate: float = 2e-5
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    logging_step: int = 2
    model_max_length: int = 2048
    weight_decay: float = 0.01
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
    )
    lora_modules_to_save: str = "embed_tokens,lm_head"
    resume_from_checkpoint: str | None = None
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine"
    save_and_eval_strategy: str = "epoch"
    save_and_eval_steps: int = 1000
    fp16: bool = False
    bf16: bool = False
    deepspeed: str = "./config/ds_z3_bf16.json"
    device: str = "cuda"


@dataclass
class TestArgs:
    """
    测试和评估相关的配置

    Attributes
    ----------
        filter_items (bool): 在评估时是否过滤掉不在候选集中的非法物品.
        results_file (str): 保存测试结果的文件路径.
        test_batch_size (int): 测试时的批处理大小.
        num_beams (int): Beam search的束宽.
        sample_num (int): 测试样本数量, -1表示使用所有测试数据.
        gpu_id (int): 单GPU测试时使用的GPU ID.
        test_prompt_ids (str): 测试时使用的prompt ID, 用逗号分隔, "all"表示全部使用.
        metrics (str): 评估指标, 用逗号分隔.
        test_task (str): 要执行的测试任务名称.

    """

    filter_items: bool = False
    results_file: str = "./results/test-ddp.json"
    test_batch_size: int = 1
    num_beams: int = 20
    sample_num: int = -1
    gpu_id: int = 0
    test_prompt_ids: str = "0"
    metrics: str = "hit@1,hit@5,hit@10,ndcg@5,ndcg@10"
    test_task: str = "SeqRec"
    models: list[BenchmarkModelArgs] = field(default_factory=list)
    # 用于文本生成 benchmark 的评估指标, 逗号分隔
    # 可选: "bleu", "rouge", "bert_score", "semantic_similarity"
    benchmark_metrics: str = "bleu,rouge"
    # --- seqrec 评测专用参数 ---
    use_constrained_generation: bool = False
    max_new_tokens: int = 10
    print_freq: int = 4


@dataclass
class TextGenerationArgs:
    """
    文本生成任务相关的配置

    Attributes
    ----------
        output_file (str): 生成结果的输出文件路径.
        max_new_tokens (int): 模型一次生成的新token的最大数量.
        temperature (float):采样温度, 用于控制生成的随机性. 较低的值更具确定性.
        top_p (float): Top-p (nucleus) 采样概率.
        do_sample (bool): 是否使用采样策略. False表示使用贪心解码.
        sample_num (int): 从数据集中选择多少样本进行生成, -1表示全部.

    """

    output_file: str = "generated_text_results.json"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    sample_num: int = -1


@dataclass
class Args:
    """统一所有配置的容器"""

    global_args: GlobalArgs = field(default_factory=GlobalArgs)
    dataset_args: DatasetArgs = field(default_factory=DatasetArgs)
    train_args: TrainingArgs = field(default_factory=TrainingArgs)
    test_args: TestArgs = field(default_factory=TestArgs)
    text_generation_args: TextGenerationArgs = field(
        default_factory=TextGenerationArgs
    )
