"""
Text Generation Quality Evaluation Benchmark (简化版)
用于评估多模态模型生成文本质量的benchmark系统
"""

import json
import logging
from typing import Any

# 核心评估指标所需的库
import nltk
import numpy as np
import pandas as pd
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# 下载必要的NLTK数据
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class TextGenerationBenchmark:
    """
    文本生成质量评估benchmark
    """

    def __init__(self, reference_data_path: str):
        """
        初始化benchmark

        Args:
        ----
            reference_data_path: 参考数据(ground truth)的路径

        """
        self.reference_data_path = reference_data_path

        # 加载参考数据
        self.reference_data = self._load_reference_data()

        # 初始化评估器
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # 用于语义相似度

        # 平滑函数用于BLEU计算
        self.smoothing = SmoothingFunction().method1

        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_reference_data(self) -> dict[str, Any]:
        """加载参考数据(ground truth)"""
        with open(self.reference_data_path, encoding="utf-8") as f:
            return json.load(f)

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
        P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)

        return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item(),
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

    def evaluate_model(
        self, model_name: str, result_file_path: str
    ) -> dict[str, Any]:
        """
        评估单个模型的生成质量

        Args:
        ----
            model_name: 模型名称
            result_file_path: 模型生成结果文件路径

        Returns:
        -------
            评估结果字典

        """
        # 加载模型结果
        with open(result_file_path, encoding="utf-8") as f:
            model_results = json.load(f)

        scores = {
            "model_name": model_name,
            "bleu_scores": [],
            "rouge_scores": {"rouge1": [], "rouge2": [], "rougeL": []},
            "semantic_similarities": [],
            "bert_references": [],
            "bert_candidates": [],
        }

        # 遍历每个物品进行评估
        for item_id, generated_text in model_results.items():
            if item_id not in self.reference_data:
                self.logger.warning(
                    f"Item {item_id} not found in reference data"
                )
                continue

            # 构建参考文本
            item_data = self.reference_data[item_id]
            reference_text = self._build_reference_text(item_data)

            # 计算各项指标
            bleu = self.calculate_bleu_score(reference_text, generated_text)
            rouge = self.calculate_rouge_scores(reference_text, generated_text)
            semantic_sim = self.calculate_semantic_similarity(
                reference_text, generated_text
            )

            scores["bleu_scores"].append(bleu)
            scores["rouge_scores"]["rouge1"].append(rouge["rouge1"])
            scores["rouge_scores"]["rouge2"].append(rouge["rouge2"])
            scores["rouge_scores"]["rougeL"].append(rouge["rougeL"])
            scores["semantic_similarities"].append(semantic_sim)

            # 收集BERTScore计算所需的数据
            scores["bert_references"].append(reference_text)
            scores["bert_candidates"].append(generated_text)

        # 计算BERTScore (批量计算更高效)
        if scores["bert_references"]:
            bert_scores = self.calculate_bert_score(
                scores["bert_references"], scores["bert_candidates"]
            )
            scores.update(bert_scores)

        # 计算平均分数
        avg_scores = {
            "model_name": model_name,
            "avg_bleu": np.mean(scores["bleu_scores"]),
            "avg_rouge1": np.mean(scores["rouge_scores"]["rouge1"]),
            "avg_rouge2": np.mean(scores["rouge_scores"]["rouge2"]),
            "avg_rougeL": np.mean(scores["rouge_scores"]["rougeL"]),
            "avg_semantic_similarity": np.mean(scores["semantic_similarities"]),
            "sample_count": len(scores["bleu_scores"]),
        }

        # 添加BERTScore
        for key in ["bert_precision", "bert_recall", "bert_f1"]:
            if key in scores:
                avg_scores[f"avg_{key}"] = scores[key]

        return avg_scores

    def compare_models(self, model_results: dict[str, str]) -> pd.DataFrame:
        """
        对比多个模型的生成质量

        Args:
        ----
            model_results: 模型名称到结果文件路径的映射 {model_name: result_file_path}

        Returns:
        -------
            包含所有模型评估结果的DataFrame

        """
        all_results = []

        for model_name, result_file in model_results.items():
            self.logger.info(f"Evaluating model: {model_name}")

            try:
                eval_results = self.evaluate_model(model_name, result_file)
                all_results.append(eval_results)
                self.logger.info(f"Completed evaluation for {model_name}")

            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e!s}")
                continue

        # 转换为DataFrame并按BLEU分数排序
        results_df = pd.DataFrame(all_results)
        if not results_df.empty:
            results_df = results_df.sort_values("avg_bleu", ascending=False)

        return results_df

    def print_results(self, results_df: pd.DataFrame):
        """打印评估结果"""
        print("\n" + "=" * 80)
        print("TEXT GENERATION QUALITY BENCHMARK RESULTS")
        print("=" * 80)

        print(
            f"{'Model':<25} {'BLEU':<8} {'ROUGE-1':<9} {'ROUGE-L':<9} {'Semantic':<9} {'BERTScore':<10} {'Samples'}"
        )
        print("-" * 80)

        for _, row in results_df.iterrows():
            bert_score = (
                f"{row.get('avg_bert_f1', 0):.4f}"
                if "avg_bert_f1" in row
                else "N/A"
            )
            print(
                f"{row['model_name']:<25} "
                f"{row['avg_bleu']:<8.4f} "
                f"{row['avg_rouge1']:<9.4f} "
                f"{row['avg_rougeL']:<9.4f} "
                f"{row['avg_semantic_similarity']:<9.4f} "
                f"{bert_score:<10} "
                f"{row['sample_count']}"
            )

    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """保存评估结果到CSV文件"""
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")
