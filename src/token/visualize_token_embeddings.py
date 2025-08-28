import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

# Optional plotly support for interactive visualizations
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


class EmbeddingInfo:
    """Container for embedding information."""

    def __init__(
        self,
        embeddings: np.ndarray,
        token_names: list[str],
        token_ids: list[int],
    ):
        self.embeddings = embeddings
        self.token_names = token_names
        self.token_ids = token_ids


def parse_token_category(token_name: str) -> str:
    """Parse token name and return its category."""
    token_name = token_name.strip()

    if token_name.startswith("<") and token_name.endswith(">"):
        inner = token_name[1:-1].strip()

        # Check for main categories a, b, c, d
        if any(inner.startswith(f"{cat}_") for cat in "abcd"):
            category = inner[0]
            return f"Category {category.upper()}"

        # Check for special tokens
        if any(keyword in inner.lower() for keyword in ["end", "start", "pad"]):
            return "Special Token"

        return "Other Format"

    return "Unknown"


def sort_tokens_by_category(
    token_names: list[str],
) -> tuple[list[str], list[int]]:
    """Sort tokens by their category."""
    category_priority = {
        "Category A": 1,
        "Category B": 2,
        "Category C": 3,
        "Category D": 4,
        "Special Token": 5,
        "Other Format": 6,
        "Unknown": 7,
    }

    def get_sort_key(token_name):
        category = parse_token_category(token_name)
        priority = category_priority.get(category, 999)

        if category.startswith("Category "):
            try:
                inner = token_name[1:-1].strip()
                if "_" in inner:
                    num_part = inner.split("_")[1]
                    num = int(num_part)
                    return (priority, num)
            except (ValueError, IndexError):
                pass

        return (priority, 0)

    token_with_indices = [(name, i) for i, name in enumerate(token_names)]
    sorted_tokens_with_indices = sorted(
        token_with_indices, key=lambda x: get_sort_key(x[0])
    )

    sorted_token_names = [item[0] for item in sorted_tokens_with_indices]
    sorted_indices = [item[1] for item in sorted_tokens_with_indices]

    return sorted_token_names, sorted_indices


def get_token_colors(
    token_names: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Generate color mapping for token categories."""
    category_colors = {
        "Category A": "#FF6B6B",  # Red
        "Category B": "#4ECDC4",  # Cyan
        "Category C": "#45B7D1",  # Blue
        "Category D": "#96CEB4",  # Green
        "Special Token": "#FFEAA7",  # Yellow
        "Other Format": "#DDA0DD",  # Purple
        "Unknown": "#A8A8A8",  # Gray
    }

    colors = []
    for token_name in token_names:
        category = parse_token_category(token_name)
        color = category_colors.get(category, "#808080")
        colors.append(color)

    return colors, category_colors


def load_safetensor_embeddings(
    safetensor_path: str, layer_name: str = "model.embed_tokens.weight"
) -> torch.Tensor:
    """Load embeddings from a safetensor file."""
    from safetensors.torch import safe_open

    with safe_open(safetensor_path, framework="pt") as f:
        embeddings = f.get_tensor(layer_name)

    if embeddings.dtype == torch.bfloat16:
        embeddings = embeddings.to(torch.float32)

    return embeddings


def load_embeddings_from_model(
    model_path: str,
    layer_name: str = "model.embed_tokens.weight",
    sample_original_tokens: int = 1000,
    seed: int = 42,
) -> EmbeddingInfo:
    """Load embeddings from model, including new tokens and sampled original tokens."""
    # Check for safetensor file
    safetensor_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensor_path):
        safetensor_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]
        if safetensor_files:
            safetensor_path = os.path.join(model_path, safetensor_files[0])
        else:
            raise FileNotFoundError(
                f"No safetensor files found in {model_path}"
            )

    print(f"Loading embeddings from: {safetensor_path}")
    embeddings = load_safetensor_embeddings(safetensor_path, layer_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = len(tokenizer)

    # Get new tokens (added tokens)
    new_token_ids = list(tokenizer.added_tokens_decoder.keys())

    # Filter for specific categories if needed
    filtered_new_tokens = [
        token_id
        for token_id in new_token_ids
        if parse_token_category(tokenizer.decode(token_id))
        in ["Category A", "Category B", "Category C", "Category D"]
    ]

    if filtered_new_tokens:
        new_token_ids = filtered_new_tokens
        print(f"Found {len(new_token_ids)} new tokens in categories A-D")
    else:
        print(f"Found {len(new_token_ids)} new tokens (all categories)")

    # Sample original tokens
    original_token_ids = list(range(min(vocab_size, embeddings.shape[0])))
    original_token_ids = [
        tid for tid in original_token_ids if tid not in new_token_ids
    ]

    random.seed(seed)
    num_samples = min(sample_original_tokens, len(original_token_ids))
    sampled_original_ids = (
        random.sample(original_token_ids, num_samples)
        if num_samples > 0
        else []
    )

    print(f"Sampled {len(sampled_original_ids)} original tokens")

    # Combine token IDs
    all_token_ids = new_token_ids + sampled_original_ids

    # Get token names
    token_names = []
    for token_id in all_token_ids:
        try:
            name = tokenizer.decode(token_id)
            # Mark original tokens
            if token_id in sampled_original_ids:
                name = f"[ORIG] {name}"
            token_names.append(name)
        except Exception:
            token_names.append(f"<token_{token_id}>")

    # Extract embeddings
    selected_embeddings = embeddings[all_token_ids, :].numpy()

    return EmbeddingInfo(selected_embeddings, token_names, all_token_ids)


def load_embeddings_from_pkl(pkl_path: str) -> EmbeddingInfo:
    """Load embeddings from a pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return EmbeddingInfo(
        embeddings=data["embeddings"],
        token_names=data["token_names"],
        token_ids=data.get("token_ids", list(range(len(data["token_names"])))),
    )


def perform_dimension_reduction(
    embeddings: np.ndarray, method: str, args: argparse.Namespace
) -> tuple[np.ndarray, str]:
    """Perform dimension reduction using t-SNE or PCA."""
    # Set environment variables for stability
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Performing {method.upper()} dimension reduction...")
    print(f"Input shape: {embeddings.shape}")

    if method == "tsne":
        # For large datasets, pre-reduce with PCA
        if embeddings.shape[0] > args.tsne_threshold:
            print(
                f"Dataset is large, pre-reducing with PCA to {args.pca_dim} dimensions..."
            )
            pca_pre = PCA(n_components=args.pca_dim, random_state=42)
            embeddings_pca = pca_pre.fit_transform(embeddings)

            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(args.perplexity, len(embeddings_pca) - 1),
                method="barnes_hut"
                if embeddings_pca.shape[0] > 5000
                else "exact",
            )
            embeddings_2d = tsne.fit_transform(embeddings_pca)
            title = f"t-SNE Visualization (PCA pre-reduced to {args.pca_dim}D)"
        else:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(args.perplexity, len(embeddings) - 1),
            )
            embeddings_2d = tsne.fit_transform(embeddings)
            title = "t-SNE Visualization"
    else:  # PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        title = f"PCA Visualization (Explained Variance: {explained_var:.3f})"

    return embeddings_2d, title


def create_static_visualization(
    embeddings_2d: np.ndarray,
    token_names: list[str],
    title: str,
    output_path: str,
    args: argparse.Namespace,
):
    """Create static matplotlib visualization."""
    plt.figure(figsize=(args.fig_width, args.fig_height))

    # Get colors
    token_colors, category_colors = get_token_colors(token_names)

    # Create scatter plot
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        alpha=args.point_alpha,
        s=args.point_size,
        c=token_colors,
    )

    # Add labels for a subset of points
    if args.show_labels:
        label_interval = max(1, len(token_names) // args.max_labels)
        for i, token_name in enumerate(token_names):
            if i % label_interval == 0:
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
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    # Create legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=category,
        )
        for category, color in category_colors.items()
        if any(parse_token_category(name) == category for name in token_names)
    ]

    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to: {output_path}")


def create_interactive_visualization(
    embeddings_2d: np.ndarray,
    token_names: list[str],
    title: str,
    output_path: str,
    args: argparse.Namespace,
):
    """Create interactive plotly visualization."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return

    fig = go.Figure()

    # Get colors
    _, category_colors = get_token_colors(token_names)

    # Group tokens by category
    categories = {}
    for i, name in enumerate(token_names):
        category = parse_token_category(name)
        if category not in categories:
            categories[category] = []
        categories[category].append(i)

    # Add traces for each category
    for category, indices in categories.items():
        color = category_colors.get(category, "#808080")

        fig.add_trace(
            go.Scatter(
                x=[embeddings_2d[i, 0] for i in indices],
                y=[embeddings_2d[i, 1] for i in indices],
                mode="markers",
                name=category,
                marker=dict(size=10, color=color, opacity=0.7),
                text=[token_names[i] for i in indices],
                hovertemplate="<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=args.interactive_width,
        height=args.interactive_height,
        showlegend=True,
    )

    fig.write_html(output_path)
    print(f"Saved interactive visualization to: {output_path}")


def analyze_embeddings(
    embeddings: np.ndarray,
    token_names: list[str],
    output_dir: str,
    args: argparse.Namespace,
):
    """Analyze embedding statistics and create similarity matrix."""
    print("\n=== Embedding Statistics ===")
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    print(f"Min: {embeddings.min():.6f}")
    print(f"Max: {embeddings.max():.6f}")

    if not args.skip_similarity:
        print("\nComputing similarity matrix...")

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normalized = embeddings / norms

        similarity_matrix = np.dot(normalized, normalized.T)
        similarity_matrix = np.nan_to_num(
            similarity_matrix, nan=0.0, posinf=1.0, neginf=-1.0
        )

        # Sort by category
        sorted_names, sorted_indices = sort_tokens_by_category(token_names)
        sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]

        # Create heatmap
        plt.figure(figsize=(args.heatmap_size, args.heatmap_size))

        # Determine tick labels
        if len(sorted_names) > 50:
            tick_labels = False  # Don't show labels if too many
        else:
            tick_labels = [
                name[:15] + "..." if len(name) > 15 else name
                for name in sorted_names
            ]

        sns.heatmap(
            sorted_similarity,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"label": "Cosine Similarity"},
        )

        plt.title(
            "Token Embedding Similarity Matrix", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        output_path = os.path.join(output_dir, "similarity_matrix.png")
        plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved similarity matrix to: {output_path}")

        # Find most/least similar pairs
        np.fill_diagonal(sorted_similarity, -1)
        max_idx = np.unravel_index(
            sorted_similarity.argmax(), sorted_similarity.shape
        )
        min_idx = np.unravel_index(
            sorted_similarity.argmin(), sorted_similarity.shape
        )

        print(
            f"\nMost similar pair: {sorted_names[max_idx[0]]} - {sorted_names[max_idx[1]]} "
            f"(similarity: {sorted_similarity[max_idx]:.4f})"
        )
        print(
            f"Least similar pair: {sorted_names[min_idx[0]]} - {sorted_names[min_idx[1]]} "
            f"(similarity: {sorted_similarity[min_idx]:.4f})"
        )


def print_category_stats(token_names: list[str]):
    """Print token category statistics."""
    print("\n=== Token Category Statistics ===")

    category_counts = {}
    for name in token_names:
        # Handle original token markers
        if name.startswith("[ORIG]"):
            category = "Original Tokens"
        else:
            category = parse_token_category(name)
        category_counts[category] = category_counts.get(category, 0) + 1

    # Sort by count
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )

    for category, count in sorted_categories:
        print(f"{category}: {count} tokens")

    print(f"Total: {len(token_names)} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize token embeddings with t-SNE/PCA"
    )

    # Input/Output arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path or embedding file (.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to model_path)",
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="model.embed_tokens.weight",
        help="Layer name for safetensor loading",
    )

    # Sampling arguments
    parser.add_argument(
        "--sample_original",
        type=int,
        default=1000,
        help="Number of original tokens to sample (default: 1000)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum total tokens to process",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )

    # Visualization method arguments
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["tsne", "pca", "both"],
        help="Dimension reduction method",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive visualizations",
    )

    # t-SNE specific arguments
    parser.add_argument(
        "--perplexity", type=int, default=30, help="t-SNE perplexity parameter"
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="PCA dimensions for t-SNE pre-reduction",
    )
    parser.add_argument(
        "--tsne_threshold",
        type=int,
        default=10000,
        help="Sample count threshold for PCA pre-reduction",
    )

    # Visualization appearance arguments
    parser.add_argument(
        "--fig_width", type=int, default=15, help="Figure width in inches"
    )
    parser.add_argument(
        "--fig_height", type=int, default=12, help="Figure height in inches"
    )
    parser.add_argument(
        "--point_size", type=int, default=100, help="Scatter plot point size"
    )
    parser.add_argument(
        "--point_alpha",
        type=float,
        default=0.7,
        help="Scatter plot point transparency",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    parser.add_argument(
        "--show_labels", action="store_true", help="Show token labels on plot"
    )
    parser.add_argument(
        "--max_labels",
        type=int,
        default=50,
        help="Maximum number of labels to show",
    )

    # Interactive visualization arguments
    parser.add_argument(
        "--interactive_width",
        type=int,
        default=1200,
        help="Interactive plot width in pixels",
    )
    parser.add_argument(
        "--interactive_height",
        type=int,
        default=800,
        help="Interactive plot height in pixels",
    )

    # Analysis arguments
    parser.add_argument(
        "--skip_analysis", action="store_true", help="Skip statistical analysis"
    )
    parser.add_argument(
        "--skip_similarity",
        action="store_true",
        help="Skip similarity matrix computation",
    )
    parser.add_argument(
        "--heatmap_size",
        type=int,
        default=14,
        help="Similarity heatmap size in inches",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.model_path
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from: {args.model_path}")

    if args.model_path.endswith(".pkl"):
        embedding_info = load_embeddings_from_pkl(args.model_path)
    else:
        embedding_info = load_embeddings_from_model(
            args.model_path, args.layer_name, args.sample_original, args.seed
        )

    embeddings = embedding_info.embeddings
    token_names = embedding_info.token_names

    # Apply max_tokens limit if specified
    if args.max_tokens and len(token_names) > args.max_tokens:
        print(f"Limiting tokens from {len(token_names)} to {args.max_tokens}")
        random.seed(args.seed)
        indices = random.sample(range(len(token_names)), args.max_tokens)
        embeddings = embeddings[indices]
        token_names = [token_names[i] for i in indices]

    print(
        f"Processing {len(token_names)} tokens with {embeddings.shape[1]}-dimensional embeddings"
    )

    # Generate visualizations
    methods = ["tsne", "pca"] if args.method == "both" else [args.method]

    for method in methods:
        print(f"\n=== {method.upper()} Visualization ===")

        # Perform dimension reduction
        embeddings_2d, title = perform_dimension_reduction(
            embeddings, method, args
        )

        # Create static visualization
        output_path = os.path.join(args.output_dir, f"embeddings_{method}.png")
        create_static_visualization(
            embeddings_2d, token_names, title, output_path, args
        )

        # Create interactive visualization if requested
        if args.interactive:
            interactive_path = os.path.join(
                args.output_dir, f"embeddings_{method}_interactive.html"
            )
            create_interactive_visualization(
                embeddings_2d, token_names, title, interactive_path, args
            )

    # Perform analysis if requested
    if not args.skip_analysis:
        print("\n=== Analysis ===")
        analyze_embeddings(embeddings, token_names, args.output_dir, args)
        print_category_stats(token_names)

    print(f"\n✓ All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
