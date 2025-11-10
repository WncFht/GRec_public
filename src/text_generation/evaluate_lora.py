import argparse
import json
import logging
import os
from typing import Any

import nltk
import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.collator import UnifiedTestCollator
from src.data import TextEnrichDataset
from src.parser import parse_dataset_args, parse_global_args, parse_test_args
from src.utils import load_model_for_inference, set_seed


class TextGenerationBenchmark:
    """
    文本生成质量评估benchmark
    """

    def __init__(
        self, reference_data_path: str, metrics: str, debug: bool = False
    ):
        """
        初始化benchmark, 按需加载所需资源

        Args:
        ----
            reference_data_path: 参考数据(ground truth)的路径
            metrics (str): 需要运行的评估指标, 逗号分隔.
            debug (bool): 是否开启调试模式.

        """
        self.reference_data_path = reference_data_path
        self.debug_mode = debug

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.metrics_to_run = {
            metric.strip() for metric in metrics.split(",") if metric.strip()
        }
        self.logger.info(f"将要运行的评估指标: {self.metrics_to_run}")

        print("正在按需初始化评估器...")
        if "rouge" in self.metrics_to_run:
            print(" -> ROUGE Scorer")
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

        if "bleu" in self.metrics_to_run:
            print(" -> NLTK (用于 BLEU)")
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")
            self.smoothing = SmoothingFunction().method1

        if "semantic_similarity" in self.metrics_to_run:
            print(" -> SentenceTransformer (用于 semantic_similarity)")
            from sentence_transformers import SentenceTransformer

            self.sentence_model = SentenceTransformer(
                "./ckpt/sentence-transformers/all-MiniLM-L6-v2"
            )
        if "bert_score" in self.metrics_to_run:
            print(" -> BERTScore (模型将在首次使用时按需加载)")

        # 加载参考数据集的所有信息，以便按ID查找
        print("正在加载完整参考数据...")
        self._load_full_reference_data()
        print("初始化完成。")

    def _load_full_reference_data(self):
        """加载完整的参考数据，用于按ID查找。"""
        # reference_data_path 样例: './data/Instruments/Instruments.item_enriched.json'
        with open(self.reference_data_path, encoding="utf-8") as f:
            # 假设数据是 {item_id: {details...}} 的格式
            self.full_reference_data = json.load(f)

    def _get_reference_item_by_id(self, item_id: str) -> dict | None:
        """根据物品ID从加载的数据中获取参考项。"""
        return self.full_reference_data.get(item_id)

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        计算BLEU分数

        BLEU (Bilingual Evaluation Understudy):
        - 基于n-gram精确匹配的指标，范围0-1，越高越好
        - 主要衡量生成文本与参考文本的词汇重叠程度
        - 适合评估文本的流畅性和准确性
        """
        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())

        weights = (0.25, 0.25, 0.25, 0.25)
        bleu_score = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=weights,
            smoothing_function=self.smoothing,
        )

        return bleu_score

    def calculate_rouge_scores(
        self, reference: str, candidate: str
    ) -> dict[str, float]:
        """
        计算ROUGE分数

        ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
        - 基于召回率的指标，范围0-1，越高越好
        - ROUGE-1: 基于unigram的召回率，衡量词汇重叠
        - ROUGE-2: 基于bigram的召回率，衡量局部连贯性
        - ROUGE-L: 基于最长公共子序列，衡量结构相似性
        - 特别适合评估摘要和内容覆盖度
        """
        scores = self.rouge_scorer.score(reference, candidate)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def calculate_bert_score(
        self, references: list[str], candidates: list[str]
    ) -> dict[str, float]:
        """
        计算BERTScore

        BERTScore:
        - 基于预训练BERT模型的语义相似度指标，范围通常0.8-1.0
        - 通过contextual embeddings捕获语义相似性
        - 相比传统指标能更好地处理释义和同义词
        - 包含Precision, Recall, F1三个维度
        - 适合评估语义保真度和表达多样性
        """
        p, r, f1 = bert_score(
            candidates,
            references,
            model_type="roberta-large",
            lang="en",
            verbose=False,
        )

        return {
            "bert_precision": p.mean().item(),
            "bert_recall": r.mean().item(),
            "bert_f1": f1.mean().item(),
        }

    def calculate_semantic_similarity(
        self, reference: str, candidate: str
    ) -> float:
        """
        计算语义相似度

        Semantic Similarity (基于Sentence Transformers):
        - 使用预训练的sentence embedding模型计算cosine相似度
        - 范围-1到1，越接近1越相似
        - 能够捕获深层语义相似性，不依赖词汇重叠
        - 适合评估内容的语义一致性和表达质量
        """
        embeddings = self.sentence_model.encode([reference, candidate])

        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)

    def _build_reference_text(self, item_data: dict[str, Any]) -> str:
        """
        构建参考文本
        可以根据需要调整参考文本的构建方式
        """
        text_parts = []

        if "enhanced_title" in item_data:
            text_parts.append(item_data["enhanced_title"])
        elif "title" in item_data:
            text_parts.append(item_data["title"])

        if item_data.get("highlights"):
            text_parts.extend(item_data["highlights"])

        if item_data.get("characteristics"):
            text_parts.extend(item_data["characteristics"][:3])  # 取前3个特征

        return " ".join(text_parts)

    def evaluate_model_in_memory(
        self,
        model: torch.nn.Module,
        model_name: str,
        model_type: str,
        processor,
        dataset: TextEnrichDataset,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """
        对单个模型进行流式生成和评估。
        """
        tokenizer = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            else processor
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 1. 创建 Test Collator 和 DataLoader
        test_collator = UnifiedTestCollator(
            args=args,
            processor_or_tokenizer=processor,
            model_type=model_type,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.test_batch_size,  # 使用测试的batch size
            collate_fn=test_collator,
            num_workers=8,  # 可以根据系统配置调整
        )

        scores = {
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "semantic_similarity": [],
            "bert_references": [],
            "bert_candidates": [],
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                inputs, reference_texts, item_ids = batch

                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # 3. 生成文本
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # 4. 解码生成的文本
                # outputs 包含了输入部分，需要跳过
                start_index = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, start_index:]
                generated_texts = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=False
                )

                # 【调试模式】如果开启，打印详细信息并只处理一个批次
                if self.debug_mode:
                    # 仅打印第一个样本的信息
                    print(
                        "\n"
                        + "=" * 25
                        + " DEBUG MODE: FIRST SAMPLE OF BATCH "
                        + "=" * 25
                    )
                    print(f"ITEM ID: {item_ids[0]}")
                    # 注意：input_text需要从dataloader外部获取，或者在collator中也返回
                    # 为了简化，我们直接打印解码后的输入
                    decoded_input = tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=False
                    )
                    print("-" * 70)
                    print(f"✅ MODEL INPUT (Decoded):\n{decoded_input}")
                    print("-" * 70)
                    print(
                        f"✅ MODEL OUTPUT (Generated Text):\n{generated_texts[0]}"
                    )
                    print("-" * 70)
                    print(
                        f"✅ GROUND TRUTH (Reference Text):\n{reference_texts[0]}"
                    )
                    print("=" * 70)

                # 5. 计算评估指标
                # 批量计算可以加速的指标 (Semantic Similarity)
                if "semantic_similarity" in self.metrics_to_run:
                    ref_embeddings = self.sentence_model.encode(reference_texts)
                    gen_embeddings = self.sentence_model.encode(generated_texts)
                    # 计算每对向量的余弦相似度
                    dot_products = np.sum(
                        ref_embeddings * gen_embeddings, axis=1
                    )
                    ref_norms = np.linalg.norm(ref_embeddings, axis=1)
                    gen_norms = np.linalg.norm(gen_embeddings, axis=1)
                    # 防止除以零
                    gen_norms[gen_norms == 0] = 1e-9
                    similarities = dot_products / (ref_norms * gen_norms)
                    scores["semantic_similarity"].extend(similarities.tolist())

                # 为BERTScore收集数据 (它在最后统一批量计算)
                if "bert_score" in self.metrics_to_run:
                    scores["bert_references"].extend(reference_texts)
                    scores["bert_candidates"].extend(generated_texts)

                # 逐个样本计算剩余的指标 (BLEU, ROUGE)
                for ref_text, gen_text in zip(
                    reference_texts, generated_texts, strict=False
                ):
                    if "bleu" in self.metrics_to_run:
                        bleu = self.calculate_bleu_score(ref_text, gen_text)
                        scores["bleu"].append(bleu)

                    if "rouge" in self.metrics_to_run:
                        rouge = self.calculate_rouge_scores(ref_text, gen_text)
                        scores["rouge1"].append(rouge["rouge1"])
                        scores["rouge2"].append(rouge["rouge2"])
                        scores["rougeL"].append(rouge["rougeL"])

                if self.debug_mode:
                    break  # 调试模式下处理完一个批次后即退出循环

        # 批量计算BERTScore
        if "bert_score" in self.metrics_to_run and scores["bert_references"]:
            bert_scores = self.calculate_bert_score(
                scores["bert_references"], scores["bert_candidates"]
            )
            scores.update(bert_scores)

        # 计算平均分
        avg_scores = {"model_name": model_name}

        if "bleu" in self.metrics_to_run:
            avg_scores["avg_bleu"] = np.mean(scores["bleu"])
        if "rouge" in self.metrics_to_run:
            avg_scores["avg_rouge1"] = np.mean(scores["rouge1"])
            avg_scores["avg_rouge2"] = np.mean(scores["rouge2"])
            avg_scores["avg_rougeL"] = np.mean(scores["rougeL"])
        if "semantic_similarity" in self.metrics_to_run:
            avg_scores["avg_semantic_similarity"] = np.mean(
                scores["semantic_similarity"]
            )
        if "bert_score" in self.metrics_to_run and "bert_f1" in scores:
            avg_scores["avg_bert_precision"] = scores["bert_precision"]
            avg_scores["avg_bert_recall"] = scores["bert_recall"]
            avg_scores["avg_bert_f1"] = scores["bert_f1"]

        avg_scores["sample_count"] = (
            len(scores["bleu"])
            if "bleu" in self.metrics_to_run
            else len(scores["rouge1"])
            if "rouge" in self.metrics_to_run
            else 0
        )

        return avg_scores

    def evaluate_single_model(self, args: argparse.Namespace) -> pd.DataFrame:
        """
        评估单个模型的生成质量

        Args:
        ----
            args: 统一的 Args Dataclass 配置对象

        Returns:
        -------
            包含单个模型评估结果的DataFrame

        """
        # 从参数中获取单个模型配置
        model_name = args.model_name
        model_type = args.model_type
        ckpt_path = args.ckpt_path
        use_lora = args.lora

        self.logger.info(f"--- Running benchmark for: {model_name} ---")
        self.logger.info(f"Using LoRA: {use_lora}")
        if use_lora:
            self.logger.info(f"Base model: {args.base_model}")

        try:
            # 1. 加载模型
            self.logger.info(f"Loading model '{model_name}'...")
            model, processor = load_model_for_inference(
                model_type=model_type,
                ckpt_path=ckpt_path,
                use_lora=use_lora,
                model_path=args.base_model if use_lora else None,
            )

            # 确保模型在正确的设备上
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            if not hasattr(model, "device"):
                model.to(device)

            # 2. 加载数据集
            self.logger.info("Loading test dataset for text enrichment...")
            dataset = TextEnrichDataset(
                args=args,
                mode="test",  # 始终使用测试集进行评估
            )

            # 3. 运行评估
            eval_results = self.evaluate_model_in_memory(
                model, model_name, model_type, processor, dataset, args
            )
            eval_results["model_name"] = model_name
            eval_results["is_lora"] = use_lora
            if use_lora:
                eval_results["base_model"] = args.base_model
            self.logger.info(f"--- Finished benchmark for: {model_name} ---")

        except Exception:
            self.logger.exception("Error evaluating")
            return pd.DataFrame()

        # 转换为DataFrame
        results_df = pd.DataFrame([eval_results])
        return results_df

    def print_results(self, results_df: pd.DataFrame):
        """打印评估结果"""
        print("\n" + "=" * 80)
        print("TEXT GENERATION QUALITY BENCHMARK RESULTS")
        if "is_lora" in results_df.columns and results_df["is_lora"].any():
            print("(LoRA Model)")
        print("=" * 80)

        # 构建动态表头
        header = f"{'Model':<25} "
        if "avg_bleu" in results_df.columns:
            header += f"{'BLEU':<8} "
        if "avg_rouge1" in results_df.columns:
            header += f"{'ROUGE-1':<9} "
        if "avg_rougeL" in results_df.columns:
            header += f"{'ROUGE-L':<9} "
        if "avg_semantic_similarity" in results_df.columns:
            header += f"{'Semantic':<9} "
        if "avg_bert_f1" in results_df.columns:
            header += f"{'BERTScore':<10} "
        header += "Samples"
        print(header)
        print("-" * len(header))

        for _, row in results_df.iterrows():
            row_str = f"{row['model_name']:<25} "
            if "avg_bleu" in row:
                row_str += f"{row['avg_bleu']:<8.4f} "
            if "avg_rouge1" in row:
                row_str += f"{row['avg_rouge1']:<9.4f} "
            if "avg_rougeL" in row:
                row_str += f"{row['avg_rougeL']:<9.4f} "
            if "avg_semantic_similarity" in row:
                row_str += f"{row['avg_semantic_similarity']:<9.4f} "
            if "avg_bert_f1" in row:
                bert_score = f"{row.get('avg_bert_f1', 0):.4f}"
                row_str += f"{bert_score:<10} "
            row_str += f"{row['sample_count']}"
            print(row_str)

    def save_results(self, results_df: pd.DataFrame, file_path: str):
        """
        将评估结果保存到CSV文件。
        如果文件已存在，则加载现有数据，更新或追加新结果。
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 添加LoRA相关信息到结果中
        results_json = results_df.to_dict("records")[0]

        # 保存为JSON格式以保持与seqrec的一致性
        json_path = file_path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=4)

        self.logger.info(f"结果已成功保存到: {json_path}")

        # 也保存CSV格式
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            self.logger.info(
                f"发现已存在的结果文件: {file_path}。正在合并结果..."
            )
            try:
                # 1. 加载旧数据
                existing_df = pd.read_csv(file_path)

                # 2. 设置 'model_name' 为索引，方便合并
                results_df = results_df.set_index("model_name")
                existing_df = existing_df.set_index("model_name")

                # 3. 更新或追加新数据
                #    - 对于已存在的模型，新结果会覆盖旧结果
                #    - 对于新模型，结果会被添加
                combined_df = existing_df.copy()
                combined_df.update(results_df)

                # 将新模型的结果（未在combined_df中更新的）追加进去
                new_models_df = results_df[
                    ~results_df.index.isin(existing_df.index)
                ]
                final_df = pd.concat([combined_df, new_models_df])

                # 4. 恢复索引
                final_df.reset_index(inplace=True)

            except pd.errors.EmptyDataError:
                self.logger.warning(
                    f"结果文件 {file_path} 为空, 将直接写入新数据。"
                )
                final_df = results_df
            except Exception:
                self.logger.exception("合并结果时出错，将覆盖旧文件。")
                final_df = results_df
        else:
            # 文件不存在或为空，直接使用新数据
            final_df = results_df

        # 保存最终的DataFrame
        final_df.to_csv(file_path, index=False)
        self.logger.info(f"CSV结果已成功保存到: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA模型文本生成评估")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print("=" * 80)
    print("LoRA模型文本生成评估测试")
    print("=" * 80)
    print(vars(args))

    # 创建评估器
    evaluator = TextGenerationBenchmark(
        reference_data_path=args.reference_data_path,
        metrics=args.benchmark_metrics,
        debug=getattr(args, "debug", False),
    )

    # 评估单个模型
    results_df = evaluator.evaluate_single_model(args)

    # 打印结果
    if not results_df.empty:
        evaluator.print_results(results_df)

        # 保存结果
        results_file = getattr(
            args,
            "results_file",
            "./results/text_generation_lora_results.csv",
        )
        evaluator.save_results(results_df, results_file)
    else:
        print("评估失败，没有生成结果。")


if __name__ == "__main__":
    main()
