import math


def get_topk_results(
    predictions: list[str],
    scores: list[float],
    targets: list[str],
    k: int,
    all_items: list[str] | None = None,
    model_type: str = "qwen_vl",
) -> list[list[int]]:
    """
    从模型生成的 beam search 结果中，计算每个样本的 top-k 命中情况。
    函数的核心任务是将模型为每个输入样本生成的k个预测序列，
    与真实目标进行比较，并返回一个表示命中位置的列表。

    Args:
    ----
        predictions (list[str]): 模型生成的预测文本列表，总长度为 B * k (B为批次大小)。
        scores (list[float]): 与每个预测文本对应的分数，总长度为 B * k。
        targets (list[str]): 每个样本的真实目标项，长度为 B。
        k (int): beam search 的 beam 数量 (num_beams)。
        all_items (list[str] | None, optional): 候选物品全集，用于过滤无效预测。默认为 None。
        model_type (str, optional): 模型类型，用于处理不同的输出格式。默认为 "qwen_vl"。

    Returns:
    -------
        list[list[int]]: 一个列表，每个元素是对应样本的 top-k 命中列表（长度为k）。
                         例如 [[0, 1, 0, ...], [1, 0, 0, ...]]，1表示在排序后的该位置命中，0表示未命中。

    """
    results = []
    B = len(targets)  # B 是批次大小 (batch size)

    # 1. 清理预测文本，去除模型输出中多余的前缀和空格
    if model_type == "qwen_vl":
        predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]

    # 2. (可选) 过滤不在候选集中的预测项
    # 如果提供了候选物品全集，将不在其中的预测项分数设为极低值，以确保它们在排序中排在最后
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000  # 赋予一个极低的分数

    # 3. 对批次中的每个样本进行处理
    for b in range(B):
        # 从扁平化的列表中，提取出当前样本的 k 个预测和分数
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        # 将预测和分数配对，并按分数进行降序排序
        pairs = list(zip(batch_seqs, batch_scores, strict=False))
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        target_item = targets[b]  # 当前样本的真实目标

        # 4. 生成命中列表 (Hit List)
        # 修复：对排序后的预测进行去重，只保留第一次出现的位置
        one_results = []
        seen_preds = set()
        for sorted_pred, _ in sorted_pairs:
            # 如果预测项已经处理过，则跳过
            if sorted_pred in seen_preds:
                one_results.append(0)  # 填充0以保持列表长度
                continue

            seen_preds.add(sorted_pred)
            if sorted_pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        # 确保 one_results 长度与原始 k 一致
        while len(one_results) < k:
            one_results.append(0)

        results.append(one_results)

    return results


def get_metrics_results(
    topk_results: list[list[int]], metrics: list[str]
) -> dict[str, float]:
    """
    根据 top-k 命中结果计算指定的评估指标。

    Args:
    ----
        topk_results (list[list[int]]): `get_topk_results` 函数的输出，表示每个样本的命中情况。
        metrics (list[str]): 需要计算的指标列表，例如 ["hit@10", "ndcg@10"]。

    Returns:
    -------
        dict[str, float]: 一个字典，存储每个指标的计算总值（在整个批次上累加）。

    """
    res = {}
    # 遍历所有需要计算的指标
    for m in metrics:
        metric_name = m.lower()
        # 根据指标名称选择相应的计算函数
        if metric_name.startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif metric_name.startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"不支持的指标: {m}")

    return res


def ndcg_k(topk_results: list[list[int]], k: int) -> float:
    r"""
    计算 NDCG@k (Normalized Discounted Cumulative Gain at k)。
    NDCG 衡量排序质量，越相关的物品排在越前面，得分越高。
    公式为: DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}
    其中 rel_i 是第 i 个位置的项目的相关性（这里是1或0）。
    由于每个样本只有一个正确答案，理想DCG (IDCG) 为1，因此 NDCG@k = DCG@k。

    Args:
    ----
        topk_results (list[list[int]]): top-k 命中结果列表。
        k (int): 计算 NDCG 的截断位置。

    Returns:
    -------
        float: 整个批次的 NDCG@k 总分。

    """
    total_ndcg = 0.0
    for row in topk_results:
        # 只考虑排名前 k 的结果
        res = row[:k]
        sample_ndcg = 0.0
        for i, rel in enumerate(res):
            # 如果在位置 i+1 命中 (rel=1)，则累加其折扣增益
            if rel == 1:
                sample_ndcg += 1 / math.log2(i + 2)  # i是0-based, 排名是i+1
        total_ndcg += sample_ndcg
    return total_ndcg


def hit_k(topk_results: list[list[int]], k: int) -> float:
    """
    计算 Hit-Rate@k (命中率@k)。
    Hit-Rate@k 衡量模型在前 k 个推荐结果中，成功命中真实目标的样本比例。

    Args:
    ----
        topk_results (list[list[int]]): top-k 命中结果列表。
        k (int): 计算命中率的截断位置。

    Returns:
    -------
        float: 整个批次的总命中次数。

    """
    total_hit = 0.0
    for row in topk_results:
        # 只考虑排名前 k 的结果
        res = row[:k]
        # 如果在前 k 个结果中至少有一次命中 (sum > 0)，则认为该样本命中
        if sum(res) > 0:
            total_hit += 1
    return total_hit
