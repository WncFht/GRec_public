# 文本生成质量评估 Benchmark

## 概述

本项目用于评估多模态推荐模型在**文本丰富任务 (Text Enrichment)** 上的文本生成质量。该任务旨在根据原始图片和文本（如商品标题、描述），生成更详细、更丰富的文本属性，例如 `enhanced_title`, `tags`, `highlights` 和 `characteristics`。

评估流程包括以下几个主要步骤：
1.  **模型训练 (可选)**：如果需要，可以运行LoRA微调脚本来训练模型。
2.  **文本生成**：使用微调前后的模型（以及其他基线模型）在 `TextEnrichDataset` 上生成文本结果。
3.  **质量评估**：将生成的文本与预设的 Ground Truth (GPT-4.1 生成的数据) 进行对比，计算 BLEU、ROUGE、BERTScore 和语义相似度等指标。
4.  **结果分析**：生成评估报告，方便对比不同模型的表现。

## 文件结构

-   `main.py`: 评估主脚本，协调文本生成和评估过程。
-   `evaluate.py`: 包含各种文本生成质量评估指标的实现 (BLEU, ROUGE, BERTScore, Semantic Similarity)。
-   `generate_text.py`: 负责加载模型并执行文本生成，将结果保存为JSON文件。
-   `README.md`: 本文件，提供项目说明和运行指南。

## 如何运行评估

请确保你已经准备好数据集 (`data/Instruments/Instruments.item_enriched.json` 和其他必要的 `.json`, `.item2id` 等文件) 和预训练模型。

1.  **进入项目根目录**：
    ```bash
    cd /home/fht/src/Recommend/unifymmgrec-master
    ```

2.  **运行自动化脚本**：
    使用 `scripts/run_text_generation_benchmark.sh` 脚本来自动化整个文本生成和评估流程。在运行前，请检查并根据你的环境修改脚本中的路径和参数。

    ```bash
    bash scripts/run_text_generation_benchmark.sh
    ```

    **脚本参数说明 (`scripts/run_text_generation_benchmark.sh`)：**
    -   `export CUDA_VISIBLE_DEVICES`: 指定用于模型推理的 GPU ID (例如 `0`)。
    -   `DATASET`: 要评估的数据集名称 (例如 `Instruments`)。
    -   `DATA_PATH`: 数据集文件的根目录 (例如 `./data`)。
    -   `BASE_MODEL_NAME`: 基座模型的名称，用于构建模型路径 (例如 `Qwen2.5-VL-3B-Instruct`)。
    -   `BASE_MODEL_DIR`: 基座模型的实际路径或 Hugging Face ID。
    -   `FINETUNED_MODEL_DIR`: 微调后模型的路径。**这通常是 `multimodal_finetune_lora.py` 训练输出的检查点目录**，例如 `./ckpt/Instruments/Qwen2.5-VL-3B-Instruct/epoch_5`。
    -   `RESULTS_DIR`: 所有生成结果和评估报告的输出目录 (例如 `./results/text_generation_benchmark`)。
    -   `MODEL_TYPE`: 基座和微调模型的类型，支持 `qwen_vl`, `llama`。
    -   `LLAVA_MODEL_DIR`: (可选) 本地 LLaVA-1.5-7B 模型的路径。如果提供，脚本将自动生成其结果。
    -   `INSTRUCTBLIP_MODEL_DIR`: (可选) 本地 InstructBLIP 模型的路径。如果提供，脚本将自动生成其结果。
    -   `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `DO_SAMPLE`: 文本生成参数。
    -   `SAMPLE_NUM`: 从测试集中采样的样本数量，`-1` 表示使用所有样本。
    -   `INDEX_FILE`: 物品 token 索引文件，例如 `.index_qwen7B.json`。

    **重要提示：**
    -   脚本中 `multimodal_finetune_lora.py` 的调用默认被注释掉。如果你需要重新训练模型，请取消注释并确保路径和参数正确设置。
    -   对于 `LLaVA-1.5-7B` 和 `InstructBLIP` 等本地基线模型，现在可以通过在脚本中设置 `LLAVA_MODEL_DIR` 和 `INSTRUCTBLIP_MODEL_DIR` 路径来自动生成评估结果。
    -   对于需要 API 调用的模型（如 `GPT-4V`），你仍然需要手动运行推理，并将生成的 JSON 结果文件（例如 `gpt4v_results.json`）放置在 `RESULTS_DIR` 目录下，脚本会自动检测并将其纳入最终的对比评估中。

## 输出结果

评估结果将保存到 `$RESULTS_DIR` 目录下，主要包括：

-   `before_finetune_results.json`: 使用基座模型生成的文本结果。
-   `after_finetune_results.json`: 使用微调后的模型生成的文本结果。
-   `llava_15_7b_results.json`, `instructblip_results.json`: 如果提供了模型路径，脚本会自动生成这些结果。
-   `gpt4v_results.json`: 其他基线模型的生成结果 (需要手动放置)。
-   `benchmark_results.csv`: 汇总所有模型评估指标的 CSV 报告文件。

`benchmark_results.csv` 包含以下平均指标：

-   **BLEU**: 衡量生成文本与参考文本的 n-gram 重叠度，越高越好 (0-1)。
-   **ROUGE-1**: 基于 unigram 的召回率，衡量词汇重叠度，越高越好 (0-1)。
-   **ROUGE-L**: 基于最长公共子序列的召回率，衡量结构相似性，越高越好 (0-1)。
-   **Semantic Similarity**: 基于 Sentence Transformers 计算的余弦相似度，衡量语义相似度，越接近1越好 (-1到1)。
-   **BERTScore**: 基于 BERT 模型的语义相似度，包含 `Precision`, `Recall`, `F1` 三个维度，通常 `F1` 越高越好。

## 进一步开发

-   **添加更多基线模型**: 为其他多模态大语言模型（如原始的 BLIP2）编写 `generate_text` 兼容的推理脚本，并将其结果添加到评估中。
-   **Prompt 工程**: 探索不同的 Prompt 模板和策略，以优化文本生成质量。
-   **高级评估**: 引入人工评估或更复杂的自动评估指标。 