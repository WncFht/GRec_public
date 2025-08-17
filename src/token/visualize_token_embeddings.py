#!/usr/bin/env python3
"""
可视化新添加的 token embedding 程序
使用 t-SNE 进行降维可视化，展示新 token 在 embedding 空间中的分布

支持多种模型格式：
1. 单个 model.safetensors 文件
2. 多个分片 safetensor 文件 (如 model-00001-of-00004.safetensors)
3. PKL 格式的 embedding 文件

使用示例：
# 处理单个 safetensor 文件
python visualize_token_embeddings.py --embedding_path /path/to/model

# 处理多个分片 safetensor 文件
python visualize_token_embeddings.py --embedding_path /path/to/model --layer_name model.embed_tokens.weight

# 处理 PKL 文件
python visualize_token_embeddings.py --embedding_path embeddings.pkl

# 指定特定的 token ID
python visualize_token_embeddings.py --embedding_path /path/to/model --token_list "1000,1001,1002"

# 生成交互式可视化
python visualize_token_embeddings.py --embedding_path /path/to/model --interactive
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

# 添加 plotly 支持交互式3D可视化
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


class EmbeddingInfo:
    def __init__(
        self, embeddings: np.ndarray, token_names: list, token_ids: list
    ):
        self.embeddings = embeddings
        self.token_names = token_names
        self.token_ids = token_ids


def parse_token_category(token_name: str) -> str:
    """
    解析 token 名称，返回其类别

    Args:
        token_name: token 名称，如 "<a_12>", "<b_3>", "<c_2>", "<d_9>", "<| end |>"

    Returns:
        token 类别字符串

    """
    # 移除可能的空格
    token_name = token_name.strip()

    # 检查是否是主要类别格式 <x_y>
    if token_name.startswith("<") and token_name.endswith(">"):
        # 提取中间部分
        inner = token_name[1:-1].strip()

        # 检查是否是 a, b, c, d 类别
        if (
            inner.startswith("a_")
            or inner.startswith("b_")
            or inner.startswith("c_")
            or inner.startswith("d_")
        ):
            category = inner[0]  # 提取 a, b, c, d
            return f"Category {category.upper()}"

        # 检查是否是特殊 token
        if (
            "end" in inner.lower()
            or "start" in inner.lower()
            or "pad" in inner.lower()
        ):
            return "Special Token"

        # 其他格式的 token
        return "Other Format"

    # 不匹配任何模式的 token
    return "Unknown"


def sort_tokens_by_category(
    token_names: list[str],
) -> tuple[list[str], list[int]]:
    """
    按照类别对 token 进行排序

    Args:
        token_names: token 名称列表

    Returns:
        (排序后的 token 名称列表, 排序后的索引列表)

    """
    # 定义类别优先级
    category_priority = {
        "Category A": 1,
        "Category B": 2,
        "Category C": 3,
        "Category D": 4,
        "Special Token": 5,
        "Other Format": 6,
        "Unknown": 7,
    }

    # 为每个 token 创建排序键
    def get_sort_key(token_name):
        category = parse_token_category(token_name)
        priority = category_priority.get(category, 999)

        # 对于 a, b, c, d 类别，提取数字部分进行排序
        if category.startswith("Category "):
            try:
                # 提取 <x_y> 中的 y 部分
                inner = token_name[1:-1].strip()
                if "_" in inner:
                    num_part = inner.split("_")[1]
                    num = int(num_part)
                    return (priority, num)
                return (priority, 0)
            except (ValueError, IndexError):
                return (priority, 0)

        return (priority, 0)

    # 创建 (token_name, original_index) 的元组列表
    token_with_indices = [(name, i) for i, name in enumerate(token_names)]

    # 按排序键排序
    sorted_tokens_with_indices = sorted(
        token_with_indices, key=lambda x: get_sort_key(x[0])
    )

    # 分离排序后的 token 名称和索引
    sorted_token_names = [item[0] for item in sorted_tokens_with_indices]
    sorted_indices = [item[1] for item in sorted_tokens_with_indices]

    return sorted_token_names, sorted_indices


def get_token_colors(token_names: list[str]) -> tuple[list[str], dict]:
    """
    为 token 类别生成颜色映射

    Args:
        token_names: token 名称列表

    Returns:
        (颜色列表, 类别到颜色的映射字典)

    """
    # 定义类别颜色映射
    category_colors = {
        "Category A": "#FF6B6B",  # 红色
        "Category B": "#4ECDC4",  # 青色
        "Category C": "#45B7D1",  # 蓝色
        "Category D": "#96CEB4",  # 绿色
        # "Special Token": "#FFEAA7",  # 黄色
        # "Other Format": "#DDA0DD",  # 紫色
        # "Unknown": "#A8A8A8",  # 灰色
    }

    # 为每个 token 分配颜色
    colors = []
    for token_name in token_names:
        category = parse_token_category(token_name)
        if category in category_colors:
            color = category_colors[category]
            colors.append(color)

    return colors, category_colors


def detect_safetensor_files(model_path: str) -> tuple[bool, list[str]]:
    """
    检测模型路径中的 safetensor 文件

    Args:
        model_path: 模型路径

    Returns:
        (是否包含safetensor文件, safetensor文件列表)

    """
    if not os.path.exists(model_path):
        return False, []

    safetensor_files = []
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            safetensor_files.append(file)

    return len(safetensor_files) > 0, safetensor_files


def load_layer_from_safetensor(
    safetensors_path: str,
    layer_name: str = "model.embed_tokens.weight",
) -> torch.Tensor:
    from safetensors.torch import safe_open

    with safe_open(safetensors_path, framework="pt") as f:
        embeddings = f.get_tensor(layer_name)
    if embeddings.dtype == torch.bfloat16:
        embeddings = embeddings.to(torch.float32)
    return embeddings


def load_layer_from_multiple_safetensors(
    model_path: str,
    layer_name: str = "model.embed_tokens.weight",
) -> torch.Tensor:
    """
    从多个 safetensor 文件中加载指定层的权重

    Args:
        model_path: 模型路径
        layer_name: 要加载的层名称
        token_list: 要提取的 token ID 列表

    Returns:
        合并后的 embedding 张量

    """
    from safetensors.torch import safe_open

    # 查找所有 safetensor 文件
    safetensor_files = []
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            safetensor_files.append(file)

    if not safetensor_files:
        raise FileNotFoundError(f"在 {model_path} 中未找到 safetensor 文件")

    # 按文件名排序，确保顺序正确
    safetensor_files.sort()
    print(
        f"找到 {len(safetensor_files)} 个 safetensor 文件: {safetensor_files}"
    )

    # 检查第一个文件以获取层信息
    first_file_path = os.path.join(model_path, safetensor_files[0])
    print(f"正在检查第一个文件: {first_file_path}")

    with safe_open(first_file_path, framework="pt") as f:
        available_keys = list(f.keys())
        print(
            f"第一个文件中的可用层: {available_keys[:10]}..."
            if len(available_keys) > 10
            else f"第一个文件中的可用层: {available_keys}"
        )

        if layer_name not in f.keys():
            # 尝试查找类似的层名
            similar_keys = [
                key
                for key in available_keys
                if "embed" in key.lower() or "token" in key.lower()
            ]
            if similar_keys:
                print(
                    f"警告: 未找到 {layer_name}，但找到类似的层: {similar_keys}"
                )
                print(f"建议使用以下层名之一: {similar_keys}")
            raise KeyError(f"层 {layer_name} 在 safetensor 文件中不存在")

        # 获取第一个文件中的层
        first_embeddings = f.get_tensor(layer_name)
        if first_embeddings.dtype == torch.bfloat16:
            first_embeddings = first_embeddings.to(torch.float32)
        print(f"第一个文件中的 {layer_name} 形状: {first_embeddings.shape}")

    # 如果只有一个文件，直接返回
    if len(safetensor_files) == 1:
        return first_embeddings

    # 如果有多个文件，需要合并
    print(f"检测到多个 safetensor 文件，正在合并 {layer_name} 层...")

    # 对于 embedding 层，通常只需要第一个文件
    # 但为了通用性，我们检查所有文件并合并
    all_embeddings = []

    for file_name in safetensor_files:
        file_path = os.path.join(model_path, file_name)
        try:
            with safe_open(file_path, framework="pt") as f:
                if layer_name in f.keys():
                    embeddings = f.get_tensor(layer_name)
                    if embeddings.dtype == torch.bfloat16:
                        embeddings = embeddings.to(torch.float32)
                    all_embeddings.append(embeddings)
                    print(
                        f"从 {file_name} 加载了 {embeddings.shape} 的 {layer_name}"
                    )
                else:
                    print(f"警告: {file_name} 中未找到 {layer_name}")
        except Exception as e:
            print(f"警告: 无法从 {file_name} 加载 {layer_name}: {e}")

    if not all_embeddings:
        raise RuntimeError(f"无法从任何 safetensor 文件中加载 {layer_name}")

    # 对于 embedding 层，我们通常只需要第一个文件
    # 因为 embedding 层通常不会分片
    final_embeddings = all_embeddings[0]

    print(f"最终 embedding 形状: {final_embeddings.shape}")
    return final_embeddings


def get_topk_related_tokens(
    token_id: int, ebd: torch.Tensor, k: int = 100, exclude: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get topk related tokens given a token id
    :param token_id: token id
    :param ebd: embedding layer
    :param k: topk
    :param exclude: exclude the token itself
    :return: topk values and indices
    """
    think_tok = ebd[token_id, :]
    cos = torch.nn.CosineSimilarity(dim=1)
    sim = cos(think_tok, ebd)
    k = k + 1 if exclude else k
    topk = torch.topk(sim, k)
    if exclude:
        return topk.values[1:], topk.indices[1:]
    return topk.values, topk.indices


def tsne(ebd: torch.Tensor, pca_dim: int = None, save_path: str = None) -> None:
    from sklearn.manifold import TSNE

    if pca_dim is not None:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_dim)
        ebd = pca.fit_transform(ebd)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, metric="cosine")
    tsne_results = tsne.fit_transform(ebd)

    if save_path is not None:
        import numpy as np

        # save tsne results as npy file
        np.save(save_path, tsne_results)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1)
    # plt.show()


def load_tsne(file_path: str, verbose: bool = False):
    import numpy as np

    tsne_results = np.load(file_path)
    if verbose:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1)
        plt.show()
    return tsne_results


def analyze_tsne(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_list = list(tokenizer.added_tokens_decoder.keys())
    print(token_list)
    safetensor_path = model_name + "/model.safetensors"
    ebd = load_layer_from_safetensor(safetensor_path, token_list=token_list)
    tsne(ebd, save_path=f"{model_name.split('/')[-1]}_tsne.npy")


def load_embeddings_from_pkl(embedding_path: str) -> dict:
    """
    从 PKL 文件加载保存的 embedding 数据

    Args:
        embedding_path: embedding 文件路径

    Returns:
        包含 embedding 信息的字典

    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding 文件不存在: {embedding_path}")

    with open(embedding_path, "rb") as f:
        embedding_info = pickle.load(f)

    print(f"从 PKL 加载了 {len(embedding_info['token_names'])} 个新 token")
    print(f"Embedding 维度: {embedding_info['embeddings'].shape}")

    return embedding_info


def load_embeddings_from_safetensor(
    model_path: str,
    token_list: list = None,
    layer_name: str = "model.embed_tokens.weight",
) -> EmbeddingInfo:
    """
    从 safetensor 文件加载 embedding 数据，支持多个分片文件

    Args:
        model_path: 模型路径
        token_list: token ID 列表
        layer_name: embedding 层名称

    Returns:
        包含 embedding 信息的字典

    """
    # 检查是否存在单个 model.safetensors 文件
    single_safetensor_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_safetensor_path):
        print(f"使用单个 safetensor 文件: {single_safetensor_path}")
        embeddings = load_layer_from_safetensor(
            single_safetensor_path, layer_name
        )
    else:
        # 使用多文件加载函数
        print("检测到多个 safetensor 文件，使用多文件加载模式")
        embeddings = load_layer_from_multiple_safetensors(
            model_path, layer_name
        )

    # 加载 tokenizer 以获取 token 名称
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    token_list = list(tokenizer.added_tokens_decoder.keys())
    # only keep the <a_*> <b_*> <c_*> <d_*> tokens
    token_list = [
        token_id
        for token_id in token_list
        if parse_token_category(tokenizer.decode(token_id))
        in ["Category A", "Category B", "Category C", "Category D"]
    ]
    print(f"使用所有新增 token: {token_list}")

    # 获取 token 名称
    token_names = []
    for token_id in token_list:
        token_names.append(tokenizer.decode(token_id))

    embeddings = embeddings[token_list, :]

    embedding_info = EmbeddingInfo(embeddings, token_names, token_list)

    print(f"从 Safetensor 加载了 {len(token_names)} 个 token")
    print(f"Embedding 维度: {embeddings.shape}")

    return embedding_info


def load_embeddings(
    embedding_path: str, model_path: str = None, token_list: list = None
) -> dict:
    """
    根据文件扩展名自动选择加载方式

    Args:
        embedding_path: embedding 文件路径
        model_path: 模型路径（用于 safetensor 加载）
        token_list: token ID 列表（用于 safetensor 加载）

    Returns:
        包含 embedding 信息的字典

    """
    if embedding_path.endswith(".pkl"):
        return load_embeddings_from_pkl(embedding_path)
    if embedding_path.endswith(".safetensors") or (
        model_path
        and os.path.exists(os.path.join(model_path, "model.safetensors"))
    ):
        if model_path is None:
            raise ValueError(
                "使用 safetensor 加载方式需要提供 --model_path 参数"
            )
        return load_embeddings_from_safetensor(model_path, token_list)
    raise ValueError("不支持的文件格式，仅支持 .pkl 或 .safetensors")


def visualize_embeddings_tsne(
    embeddings_np: np.ndarray,
    token_names: list,
    output_dir: str,
    method: str = "tsne",
    args: argparse.Namespace = None,
) -> None:
    """
    使用 t-SNE 或 PCA 可视化 embedding

    Args:
        embeddings: token embedding 张量
        token_names: token 名称列表
        output_dir: 输出目录
        method: 降维方法 ("tsne" 或 "pca")

    """
    # 设置环境变量以避免 OpenBLAS 问题
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"开始 {method.upper()} 降维...")
    print(f"输入数据形状: {embeddings_np.shape}")
    print(f"内存使用量: {embeddings_np.nbytes / 1024 / 1024:.2f} MB")

    try:
        if method == "tsne":
            # 对于大数据集，先使用 PCA 降维以减少内存使用
            if embeddings_np.shape[0] > 10000:
                print(
                    f"数据量较大，先使用 PCA 降维到 {args.pca_pre_reduce} 维..."
                )
                pca_pre = PCA(n_components=args.pca_pre_reduce, random_state=42)
                embeddings_pca = pca_pre.fit_transform(embeddings_np)
                print(f"PCA 预降维后形状: {embeddings_pca.shape}")

                # 使用 t-SNE 降维到 2D
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embeddings_pca) - 1),
                    method="barnes_hut"
                    if embeddings_pca.shape[0] > 5000
                    else "exact",
                )
                embeddings_2d = tsne.fit_transform(embeddings_pca)
                title = f"t-SNE Visualization of New Token Embeddings (PCA pre-reduced to {args.pca_pre_reduce}D)"
            else:
                # 使用 t-SNE 降维到 2D
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embeddings_np) - 1),
                )
                embeddings_2d = tsne.fit_transform(embeddings_np)
                title = "t-SNE Visualization of New Token Embeddings"
        else:
            # 使用 PCA 降维到 2D
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings_np)
            title = f"PCA Visualization of New Token Embeddings (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})"

        print(f"降维完成，输出形状: {embeddings_2d.shape}")

        # 创建可视化
        plt.figure(figsize=(15, 12))

        # 获取 token 类别颜色
        token_colors, category_colors = get_token_colors(token_names)

        # 创建散点图，使用类别颜色
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.7,
            s=100,
            c=token_colors,
        )

        # 添加 token 标签（只显示部分以避免过于拥挤）
        label_interval = max(1, len(token_names) // 50)  # 最多显示 50 个标签
        for i, token_name in enumerate(token_names):
            if i % label_interval == 0:  # 只显示部分标签
                # 处理过长的 token 名称
                display_name = (
                    token_name[:20] + "..."
                    if len(token_name) > 20
                    else token_name
                )
                plt.annotate(
                    display_name,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                    ),
                )

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel(f"{method.upper()} Dimension 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Dimension 2", fontsize=12)

        # 创建自定义图例
        legend_elements = []
        for category, color in category_colors.items():
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=category,
                )
            )

        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(
            output_dir, f"new_token_embeddings_{method}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"可视化结果已保存到: {output_path}")

        # 清理内存
        plt.close()

    except Exception as e:
        print(f"降维过程中出现错误: {e}")
        print("尝试使用更保守的设置...")

        # 如果 t-SNE 失败，回退到 PCA
        if method == "tsne":
            print("回退到 PCA 降维...")
            try:
                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(embeddings_np)
                title = f"PCA Visualization (Fallback) - Explained Variance: {pca.explained_variance_ratio_.sum():.3f}"

                # 创建可视化
                plt.figure(figsize=(15, 12))

                # 获取 token 类别颜色
                token_colors, category_colors = get_token_colors(token_names)

                scatter = plt.scatter(
                    embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    alpha=0.7,
                    s=100,
                    c=token_colors,
                )

                plt.title(title, fontsize=16, fontweight="bold")
                plt.xlabel("PCA Dimension 1", fontsize=12)
                plt.ylabel("PCA Dimension 2", fontsize=12)

                # 创建自定义图例
                legend_elements = []
                for category, color in category_colors.items():
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=10,
                            label=category,
                        )
                    )

                plt.legend(
                    handles=legend_elements, loc="upper right", fontsize=10
                )

                plt.tight_layout()

                output_path = os.path.join(
                    output_dir, "new_token_embeddings_pca_fallback.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"PCA 回退可视化结果已保存到: {output_path}")
                plt.close()

            except Exception as e2:
                print(f"PCA 回退也失败了: {e2}")
                print("无法生成可视化，请检查内存或尝试减少数据量")
        else:
            print("PCA 降维失败，无法生成可视化")


def visualize_embeddings_tsne_interactive(
    embeddings_np: np.ndarray,
    token_names: list,
    output_dir: str,
    method: str = "tsne",
) -> None:
    """
    创建交互式2D可视化（使用 Plotly）

    Args:
        embeddings: token embedding 张量
        token_names: token 名称列表
        output_dir: 输出目录
        method: 降维方法 ("tsne" 或 "pca")

    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return

    # 设置环境变量以避免 OpenBLAS 问题
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"开始交互式 2D {method.upper()} 降维...")
    print(f"输入数据形状: {embeddings_np.shape}")

    try:
        if method == "tsne":
            # 对于大数据集，先使用 PCA 降维以减少内存使用
            if embeddings_np.shape[0] > 10000:
                print("数据量较大，先使用 PCA 降维到 50 维...")
                pca_pre = PCA(n_components=50, random_state=42)
                embeddings_pca = pca_pre.fit_transform(embeddings_np)
                print(f"PCA 预降维后形状: {embeddings_pca.shape}")

                # 使用 t-SNE 降维到 2D
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embeddings_pca) - 1),
                    method="barnes_hut"
                    if embeddings_pca.shape[0] > 5000
                    else "exact",
                )
                embeddings_2d = tsne.fit_transform(embeddings_pca)
                title = "2D t-SNE Interactive Visualization (PCA pre-reduced to 50D)"
            else:
                # 使用 t-SNE 降维到 2D
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embeddings_np) - 1),
                )
                embeddings_2d = tsne.fit_transform(embeddings_np)
                title = (
                    "2D t-SNE Interactive Visualization of New Token Embeddings"
                )
        else:
            # 使用 PCA 降维到 2D
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings_np)
            title = f"2D PCA Interactive Visualization of New Token Embeddings (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})"

        print(f"降维完成，输出形状: {embeddings_2d.shape}")

        # 创建交互式2D散点图
        fig = go.Figure()

        # 获取 token 类别颜色
        token_colors, category_colors = get_token_colors(token_names)

        # 为每个类别创建单独的 trace
        for category, color in category_colors.items():
            # 找到属于该类别的 token 索引
            category_indices = [
                i
                for i, name in enumerate(token_names)
                if parse_token_category(name) == category
            ]

            if category_indices:  # 只添加有数据的类别
                # 为每个点创建单独的 trace 以显示具体的 token 名称
                for idx in category_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=[embeddings_2d[idx, 0]],
                            y=[embeddings_2d[idx, 1]],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=color,
                                opacity=0.8,
                            ),
                            hovertemplate=f"<b>Token:</b> {token_names[idx]}<br>"
                            f"<b>Category:</b> {category}<br>"
                            "<b>X:</b> %{x:.3f}<br>"
                            "<b>Y:</b> %{y:.3f}<extra></extra>",
                            name=category,
                            showlegend=False,  # 避免图例重复
                        )
                    )

        # 添加图例 trace
        for category, color in category_colors.items():
            if any(
                parse_token_category(name) == category for name in token_names
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=12, color=color),
                        name=category,
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title=f"{method.upper()} Dimension 1",
            yaxis_title=f"{method.upper()} Dimension 2",
            width=1200,
            height=800,
            showlegend=True,
            margin=dict(l=50, r=50, b=50, t=50),
        )

        # 保存为HTML文件（可在浏览器中交互）
        html_path = os.path.join(
            output_dir, f"new_token_embeddings_2d_interactive_{method}.html"
        )
        fig.write_html(html_path)
        print(f"交互式2D可视化已保存到: {html_path}")
        print("在浏览器中打开此文件即可进行交互操作（悬停、缩放等）")

        # 显示图形
        # fig.show()

    except Exception as e:
        print(f"交互式降维过程中出现错误: {e}")
        print("尝试使用 PCA 降维作为回退...")

        try:
            # 回退到 PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings_np)
            title = f"2D PCA Interactive Visualization (Fallback) - Explained Variance: {pca.explained_variance_ratio_.sum():.3f}"

            # 创建交互式2D散点图
            fig = go.Figure()

            # 获取 token 类别颜色
            token_colors, category_colors = get_token_colors(token_names)

            # 为每个类别创建单独的 trace
            for category, color in category_colors.items():
                # 找到属于该类别的 token 索引
                category_indices = [
                    i
                    for i, name in enumerate(token_names)
                    if parse_token_category(name) == category
                ]

                if category_indices:  # 只添加有数据的类别
                    # 为每个点创建单独的 trace 以显示具体的 token 名称
                    for idx in category_indices:
                        fig.add_trace(
                            go.Scatter(
                                x=[embeddings_2d[idx, 0]],
                                y=[embeddings_2d[idx, 1]],
                                mode="markers",
                                marker=dict(
                                    size=12,
                                    color=color,
                                    opacity=0.8,
                                ),
                                hovertemplate=f"<b>Token:</b> {token_names[idx]}<br>"
                                f"<b>Category:</b> {category}<br>"
                                "<b>X:</b> %{x:.3f}<br>"
                                "<b>Y:</b> %{y:.3f}<extra></extra>",
                                name=category,
                                showlegend=False,  # 避免图例重复
                            )
                        )

            # 添加图例 trace
            for category, color in category_colors.items():
                if any(
                    parse_token_category(name) == category
                    for name in token_names
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(size=12, color=color),
                            name=category,
                            showlegend=True,
                            hoverinfo="skip",
                        )
                    )

            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_title="PCA Dimension 1",
                yaxis_title="PCA Dimension 2",
                width=1200,
                height=800,
                showlegend=True,
                margin=dict(l=50, r=50, b=50, t=50),
            )

            # 保存为HTML文件
            html_path = os.path.join(
                output_dir,
                "new_token_embeddings_2d_interactive_pca_fallback.html",
            )
            fig.write_html(html_path)
            print(f"PCA 回退交互式可视化已保存到: {html_path}")

        except Exception as e2:
            print(f"PCA 回退也失败了: {e2}")
            print("无法生成交互式可视化，请检查内存或尝试减少数据量")


def visualize_embeddings_3d(
    embeddings_np: np.ndarray,
    token_names: list,
    output_dir: str,
    method: str = "pca",
) -> None:
    """
    3D 可视化 embedding

    Args:
        embeddings: token embedding 张量
        token_names: token 名称列表
        output_dir: 输出目录
        method: 降维方法 ("pca" 或 "tsne")

    """
    print(f"开始 3D {method.upper()} 降维...")

    if method == "pca":
        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(embeddings_np)
        title = f"3D PCA Visualization of New Token Embeddings (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})"
    else:
        tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=min(30, len(embeddings_np) - 1),
        )
        embeddings_3d = tsne.fit_transform(embeddings_np)
        title = "3D t-SNE Visualization of New Token Embeddings"

    # 创建 3D 图
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")

    # 获取 token 类别颜色
    token_colors, category_colors = get_token_colors(token_names)

    # 创建散点图，使用类别颜色
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        alpha=0.7,
        s=100,
        c=token_colors,
    )

    # 添加 token 标签
    for i, token_name in enumerate(token_names):
        display_name = (
            token_name[:15] + "..." if len(token_name) > 15 else token_name
        )
        ax.text(
            embeddings_3d[i, 0],
            embeddings_3d[i, 1],
            embeddings_3d[i, 2],
            display_name,
            fontsize=8,
            alpha=0.8,
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    ax.set_zlabel(f"{method.upper()} Dimension 3", fontsize=12)

    # 创建自定义图例
    legend_elements = []
    for category, color in category_colors.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=category,
            )
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(
        output_dir, f"new_token_embeddings_3d_{method}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"3D 可视化结果已保存到: {output_path}")

    # plt.show()


def visualize_embeddings_3d_interactive(
    embeddings_np: np.ndarray,
    token_names: list,
    output_dir: str,
    method: str = "pca",
) -> None:
    """
    创建交互式3D可视化（使用 Plotly）

    Args:
        embeddings: token embedding 张量
        token_names: token 名称列表
        output_dir: 输出目录
        method: 降维方法 ("pca" 或 "tsne")

    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return

    print(f"开始交互式 3D {method.upper()} 降维...")

    if method == "pca":
        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(embeddings_np)
        title = f"3D PCA Interactive Visualization of New Token Embeddings (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})"
    else:
        tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=min(30, len(embeddings_np) - 1),
        )
        embeddings_3d = tsne.fit_transform(embeddings_np)
        title = "3D t-SNE Interactive Visualization of New Token Embeddings"

    # 创建交互式3D散点图
    fig = go.Figure()

    # 获取 token 类别颜色
    token_colors, category_colors = get_token_colors(token_names)

    # 为每个类别创建散点
    for category, color in category_colors.items():
        # 找到属于该类别的 token 索引
        category_indices = [
            i
            for i, name in enumerate(token_names)
            if parse_token_category(name) == category
        ]

        if category_indices:  # 只添加有数据的类别
            # 为每个点创建单独的 trace 以显示具体的 token 名称
            for idx in category_indices:
                fig.add_trace(
                    go.Scatter3d(
                        x=[embeddings_3d[idx, 0]],
                        y=[embeddings_3d[idx, 1]],
                        z=[embeddings_3d[idx, 2]],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=color,
                            opacity=0.8,
                        ),
                        name=category,
                        showlegend=False,  # 避免图例重复
                        hovertemplate=f"<b>Token:</b> {token_names[idx]}<br>"
                        f"<b>Category:</b> {category}<br>"
                        "<b>X:</b> %{x:.3f}<br>"
                        "<b>Y:</b> %{y:.3f}<br>"
                        "<b>Z:</b> %{z:.3f}<extra></extra>",
                    )
                )

    # 添加图例 trace
    for category, color in category_colors.items():
        if any(parse_token_category(name) == category for name in token_names):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=category,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    # 更新布局
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"{method.upper()} Dimension 1",
            yaxis_title=f"{method.upper()} Dimension 2",
            zaxis_title=f"{method.upper()} Dimension 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=1200,
        height=800,
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=50),
    )

    # 保存为HTML文件（可在浏览器中交互）
    html_path = os.path.join(
        output_dir, f"new_token_embeddings_3d_interactive_{method}.html"
    )
    fig.write_html(html_path)
    print(f"交互式3D可视化已保存到: {html_path}")
    print("在浏览器中打开此文件即可进行交互操作（旋转、缩放、悬停等）")

    # 显示图形
    # fig.show()


def analyze_embeddings(
    embeddings_np: np.ndarray, token_names: list, output_dir: str
) -> None:
    """
    分析 embedding 的统计特性

    Args:
        embeddings: token embedding 张量
        token_names: token 名称列表
        output_dir: 输出目录

    """
    print("\n=== Embedding 统计分析 ===")
    print(f"Embedding 形状: {embeddings_np.shape}")
    print(f"均值: {embeddings_np.mean():.6f}")
    print(f"标准差: {embeddings_np.std():.6f}")
    print(f"最小值: {embeddings_np.min():.6f}")
    print(f"最大值: {embeddings_np.max():.6f}")

    # 计算 token 之间的相似度矩阵
    print("\n计算 token 相似度矩阵...")

    # 检查是否有全零的 embedding
    # zero_embeddings = np.all(embeddings_np == 0, axis=1)
    # if np.any(zero_embeddings):
    #     print(f"警告: 发现 {np.sum(zero_embeddings)} 个全零 embedding")
    #     # 将全零 embedding 替换为很小的随机值
    #     embeddings_np[zero_embeddings] = np.random.normal(
    #         0, 1e-6, embeddings_np[zero_embeddings].shape
    #     )

    # 安全归一化，避免除零错误
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # 避免除零
    normalized_embeddings = embeddings_np / norms

    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # 处理 NaN 值
    similarity_matrix = np.nan_to_num(
        similarity_matrix, nan=0.0, posinf=1.0, neginf=-1.0
    )

    # 对 token 进行排序
    sorted_token_names, sorted_indices = sort_tokens_by_category(token_names)
    sorted_similarity_matrix = similarity_matrix[sorted_indices][
        :, sorted_indices
    ]

    # 可视化相似度矩阵
    plt.figure(figsize=(16, 14))

    # 创建类别分隔线
    category_boundaries = []
    current_category = None
    for i, token_name in enumerate(sorted_token_names):
        category = parse_token_category(token_name)
        if category != current_category:
            if current_category is not None:
                category_boundaries.append(i)
            current_category = category

    # 绘制热力图
    sns.heatmap(
        sorted_similarity_matrix,
        xticklabels=[
            t[:15] + "..." if len(t) > 15 else t for t in sorted_token_names
        ],
        yticklabels=[
            t[:15] + "..." if len(t) > 15 else t for t in sorted_token_names
        ],
        cmap="coolwarm",
        center=0,
        annot=False,
        square=True,
        cbar_kws={"label": "Cosine Similarity"},
    )

    # 添加类别分隔线
    for boundary in category_boundaries:
        plt.axhline(y=boundary, color="black", linewidth=2, alpha=0.7)
        plt.axvline(x=boundary, color="black", linewidth=2, alpha=0.7)

    plt.title(
        "Token Embedding Similarity Matrix (Sorted by Category)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Token", fontsize=12)
    plt.ylabel("Token", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # 调整布局以适应更长的标签
    plt.tight_layout()

    # 保存相似度矩阵
    output_path = os.path.join(output_dir, "token_similarity_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"相似度矩阵已保存到: {output_path}")

    # plt.show()

    # 找到最相似和最不相似的 token 对
    np.fill_diagonal(sorted_similarity_matrix, -1)  # 排除自身
    max_sim_idx = np.unravel_index(
        sorted_similarity_matrix.argmax(), sorted_similarity_matrix.shape
    )
    min_sim_idx = np.unravel_index(
        sorted_similarity_matrix.argmin(), sorted_similarity_matrix.shape
    )

    print(
        f"\n最相似的 token 对: {sorted_token_names[max_sim_idx[0]]} - {sorted_token_names[max_sim_idx[1]]} (相似度: {sorted_similarity_matrix[max_sim_idx]:.4f})"
    )
    print(
        f"最不相似的 token 对: {sorted_token_names[min_sim_idx[0]]} - {sorted_token_names[min_sim_idx[1]]} (相似度: {sorted_similarity_matrix[min_sim_idx]:.4f})"
    )

    # 打印各类别的相似度统计
    print("\n=== 各类别相似度统计 ===")
    for category in [
        "Category A",
        "Category B",
        "Category C",
        "Category D",
        "Special Token",
        "Other Format",
        "Unknown",
    ]:
        category_indices = [
            i
            for i, name in enumerate(sorted_token_names)
            if parse_token_category(name) == category
        ]
        if category_indices:
            # 计算该类别的平均相似度
            category_similarities = []
            for i in category_indices:
                for j in category_indices:
                    if i != j:
                        category_similarities.append(
                            sorted_similarity_matrix[i, j]
                        )

            if category_similarities:
                avg_sim = np.mean(category_similarities)
                print(f"{category}: 平均相似度 {avg_sim:.4f}")


def print_token_category_stats(token_names: list[str]) -> None:
    """
    打印 token 类别统计信息

    Args:
        token_names: token 名称列表

    """
    print("\n=== Token 类别统计 ===")

    category_counts = {}
    for token_name in token_names:
        category = parse_token_category(token_name)
        category_counts[category] = category_counts.get(category, 0) + 1

    # 按数量排序
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )

    for category, count in sorted_categories:
        print(f"{category}: {count} tokens")

    print(f"总计: {len(token_names)} tokens")

    # 显示一些示例
    print("\n=== 各类别示例 ===")
    for category in category_counts:
        examples = [
            name
            for name in token_names
            if parse_token_category(name) == category
        ][:3]
        print(f"{category}: {examples}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="可视化新添加的 token embedding"
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="新 token embedding 文件路径 (.pkl) 或模型路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（用于 safetensor 加载方式）",
    )
    parser.add_argument(
        "--token_list",
        type=str,
        default=None,
        help="token ID 列表，用逗号分隔（用于 safetensor 加载方式）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualization_results",
        help="可视化结果输出目录",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["tsne", "pca", "both"],
        help="降维方法: tsne, pca, 或 both",
    )
    parser.add_argument("--no_3d", action="store_true", help="不生成 3D 可视化")
    parser.add_argument(
        "--interactive", action="store_true", help="生成交互式可视化（2D和3D）"
    )
    parser.add_argument(
        "--no_analysis", action="store_true", help="不进行统计分析"
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="model.embed_tokens.weight",
        help="要加载的层名称（默认为 model.embed_tokens.weight）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="限制处理的 token 数量以减少内存使用",
    )
    parser.add_argument(
        "--pca_pre_reduce",
        type=int,
        default=50,
        help="t-SNE 前的 PCA 预降维维度（默认 50）",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_path)
    token_list = list(tokenizer.added_tokens_decoder.keys())
    print(token_list)

    args.output_dir = args.embedding_path
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 embedding 数据
    print(f"正在加载 embedding 数据: {args.embedding_path}")

    # 根据文件扩展名或参数自动选择加载方式
    has_safetensor, safetensor_files = detect_safetensor_files(
        args.embedding_path
    )

    if has_safetensor:
        if (
            len(safetensor_files) == 1
            and "model.safetensors" in safetensor_files
        ):
            print(f"检测到单个 safetensor 文件: {safetensor_files[0]}")
        else:
            print(
                f"检测到 {len(safetensor_files)} 个 safetensor 文件: {safetensor_files}"
            )
        embedding_info = load_embeddings_from_safetensor(
            args.embedding_path, token_list, args.layer_name
        )
    elif args.embedding_path.endswith(".pkl"):
        embedding_info = load_embeddings_from_pkl(args.embedding_path)
    elif args.model_path:
        has_safetensor_model, _ = detect_safetensor_files(args.model_path)
        if has_safetensor_model:
            embedding_info = load_embeddings_from_safetensor(
                args.model_path, token_list, args.layer_name
            )
        else:
            raise FileNotFoundError(
                f"在 {args.model_path} 中未找到 safetensor 文件"
            )
    # 尝试自动判断
    else:
        embedding_info = load_embeddings_from_pkl(args.embedding_path)

    embeddings_np = embedding_info.embeddings
    token_names = embedding_info.token_names
    token_lists = embedding_info.token_ids

    # 如果指定了最大 token 数量，进行采样
    if args.max_tokens and len(token_names) > args.max_tokens:
        print(f"限制 token 数量从 {len(token_names)} 到 {args.max_tokens}")
        import random

        random.seed(42)  # 确保可重复性
        indices = random.sample(range(len(token_names)), args.max_tokens)
        embeddings_np = embeddings_np[indices]
        token_names = [token_names[i] for i in indices]
        print(f"采样后的 embedding 形状: {embeddings_np.shape}")

    # 生成可视化
    if args.method in ["tsne", "both"]:
        visualize_embeddings_tsne(
            embeddings_np, token_names, args.output_dir, "tsne", args
        )
        if args.interactive:
            visualize_embeddings_tsne_interactive(
                embeddings_np, token_names, args.output_dir, "tsne"
            )

    if args.method in ["pca", "both"]:
        visualize_embeddings_tsne(
            embeddings_np, token_names, args.output_dir, "pca", args
        )
        if args.interactive:
            visualize_embeddings_tsne_interactive(
                embeddings_np, token_names, args.output_dir, "pca"
            )

    # 3D 可视化
    if not args.no_3d:
        if args.method in ["pca", "both"]:
            visualize_embeddings_3d(
                embeddings_np, token_names, args.output_dir, "pca"
            )
        if args.method in ["tsne", "both"]:
            visualize_embeddings_3d(
                embeddings_np, token_names, args.output_dir, "tsne"
            )

    # 交互式3D可视化
    if args.interactive:
        if args.method in ["pca", "both"]:
            visualize_embeddings_3d_interactive(
                embeddings_np, token_names, args.output_dir, "pca"
            )
        if args.method in ["tsne", "both"]:
            visualize_embeddings_3d_interactive(
                embeddings_np, token_names, args.output_dir, "tsne"
            )

    # 统计分析
    if not args.no_analysis:
        analyze_embeddings(embeddings_np, token_names, args.output_dir)
        print_token_category_stats(token_names)

    print(f"\n所有可视化结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
