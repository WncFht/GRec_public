import argparse

from evaluate import TextGenerationBenchmark


def main(args):
    """
    主函数：运行文本生成质量评估
    """
    domain = args.domain

    # 初始化benchmark
    benchmark = TextGenerationBenchmark(
        reference_data_path=f"data/{domain}/{domain}.item_enriched.json"
    )

    # 直接指定要对比的模型结果文件
    model_results = {
        "GPT-4V": "results/gpt4v_results.json",
        "LLaVA-1.5-7B": "results/llava_7b_results.json",
        "Before-Finetune": "results/before_finetune_results.json",
        "After-Finetune": "results/after_finetune_results.json",
        "InstructBLIP": "results/instructblip_results.json",
    }

    # 运行对比评估
    results_df = benchmark.compare_models(model_results)

    # 打印结果
    benchmark.print_results(results_df)

    # 保存结果
    benchmark.save_results(results_df, "benchmark_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Generation Quality Evaluation Benchmark"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Instruments",
        help="Dataset domain name (e.g., Arts, Games, etc.)",
    )
    args = parser.parse_args()
    main(args)
