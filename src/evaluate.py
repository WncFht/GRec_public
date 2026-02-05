import math
from collections.abc import Iterable


def clean_prediction_text(text: str) -> str:
    """
    Convert a decoded generation (often包含prompt+response) into an item string.

    This is used both by metric computation and rollout cache saving, so keep it
    lightweight and deterministic.
    """
    if not isinstance(text, str):
        text = str(text)

    # Common prompt/response delimiters.
    if "Response:" in text:
        text = text.split("Response:")[-1]
    if "assistant" in text:
        text = text.split("assistant")[-1]

    text = text.strip().replace(" ", "")
    text = text.replace("\n", "").replace("assistant", "")

    # add > to the end of the prediction if it is not there,
    # like <a_1><b_2><c_3><d_12 to <a_1><b_2><c_3><d_12>
    if text and text[0] == "<" and text[-1] != ">":
        text += ">"

    return text


def clean_predictions(predictions: Iterable[str]) -> list[str]:
    return [clean_prediction_text(_) for _ in predictions]


def get_topk_results(
    predictions, scores, targets, k, all_items=None, clean: bool = True
):
    results = []
    B = len(targets)
    if clean:
        predictions = clean_predictions(predictions)
    else:
        predictions = list(predictions)
    print()
    print(predictions[: min(k // 2, 5)])
    print([targets[0]] * min(k // 2, 5))
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores, strict=False)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit
