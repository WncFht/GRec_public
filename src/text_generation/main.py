from pathlib import Path

from ..config import parse_args
from .evaluate import TextGenerationBenchmark


def main():
    """
    主函数：基于统一配置运行文本生成和评估的 benchmark
    """
    # --- 从统一配置系统初始化 ---
    # parse_args 会自动处理 --config_file, 默认为 'config.yml'
    # 运行此脚本时, 请确保使用 --config_file 指定正确的 benchmark 配置文件, 例如:
    # python -m src.text_generation.main --config_file config/text_generation_benchmark.yml
    args = parse_args()

    # --- 准备路径和目录 ---
    # 使用 global_args.output_dir 作为结果目录
    results_dir = Path(args.global_args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 构造并检查参考数据文件路径
    reference_data_path = (
        Path(args.dataset_args.data_path)
        / args.dataset_args.dataset
        / f"{args.dataset_args.dataset}.item_enriched_v2.json"
    )
    if not reference_data_path.exists():
        print(f"错误: 找不到参考数据 {reference_data_path}")
        print("请在配置文件中检查 'dataset_args.data_path' 和 'dataset_args.dataset' 是否正确。")
        return

    # --- 运行评估 ---
    # 将整个 args 对象传递给 benchmark 类
    benchmark = TextGenerationBenchmark(
        reference_data_path=str(reference_data_path),
        metrics=args.test_args.benchmark_metrics,
        debug=args.global_args.debug,
    )

    # 动态获取配置文件路径, 以便在日志中显示
    # (parse_args 目前没有将文件名存入args对象, 我们暂时从 sys.argv 解析)
    import sys
    config_file_path = "config/text_generation_benchmark.yml" # 默认值
    for i, arg in enumerate(sys.argv):
        if arg == "--config_file" and i + 1 < len(sys.argv):
            config_file_path = sys.argv[i+1]
            break

    print(f"\n--- 使用配置文件 {config_file_path} 中的模型进行比较 ---")
    # compare_models 现在也接收整个 args 对象
    results_df = benchmark.compare_models(args)

    # --- 打印并保存结果 ---
    if not results_df.empty:
        benchmark.print_results(results_df)
        results_csv_path = results_dir / "benchmark_results.csv"
        benchmark.save_results(results_df, str(results_csv_path))
    else:
        print("评估完成，但没有生成结果。请检查日志以获取错误信息。")


if __name__ == "__main__":
    main()
