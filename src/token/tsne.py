import argparse
import json
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoTokenizer

# Optional plotly support for interactive visualizations
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


class EmbeddingInfo:
    """Container for embedding information with language detection."""

    def __init__(
        self,
        embeddings: np.ndarray,
        token_names: list[str],
        token_ids: list[int],
        token_languages: list[str] = None,
    ):
        self.embeddings = embeddings
        self.token_names = token_names
        self.token_ids = token_ids
        self.token_languages = token_languages if token_languages else []


def bytes_to_unicode() -> dict[int, str]:
    """
    Returns mapping between utf-8 bytes and unicode strings.
    Avoids mapping to whitespace/control characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs, strict=False))


def unicode_to_bytes() -> dict[str, bytes]:
    """Returns reverse mapping from unicode strings to utf-8 bytes."""
    return {v: bytes([k]) for k, v in bytes_to_unicode().items()}


def convert_to_readable_vocab(
    tokenizer, verbose: bool = False
) -> dict[int, str]:
    """Convert tokenizer vocabulary to human-readable format."""
    reversed_vocab = {v: k for k, v in tokenizer.vocab.items()}
    added_tokens = tokenizer.added_tokens_encoder
    uni2bytes_mapper = unicode_to_bytes()
    uni2bytes = lambda c: b"".join([uni2bytes_mapper[ch] for ch in c])

    readable = {}
    items = sorted(reversed_vocab.items(), key=lambda x: x[0])

    for k, v in tqdm(
        items, desc="Converting to readable vocab", disable=not verbose
    ):
        try:
            if v in added_tokens:
                readable[k] = f"ADDED_TOKEN: {v}"
            else:
                readable[k] = uni2bytes(v).decode("utf-8")
        except:
            readable[k] = f"INVALID UTF-8: {uni2bytes(v)}"

    return readable


def detect_token_language(token: str) -> str:
    """Detect the language/category of a token."""
    token_stripped = token.strip()

    # Special token patterns
    # Note: "ADDED_TOKEN:" prefix is now handled in parse_token_category
    if token.startswith("ADDED_TOKEN"):
        # This should only be reached if parse_token_category didn't handle it
        return "Added Token"
    if token.startswith("INVALID UTF-8"):
        return "Invalid UTF-8"
    if token_stripped == "":
        return "Whitespace"

    # Language patterns (ordered by specificity)
    language_patterns = {
        # Asian languages
        "Chinese": r"^[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF\U00030000-\U0003134F]+$",
        "Japanese": r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u303B\u309D\u30FD]+$",
        "Korean": r"^[\uAC00-\uD7A3]+$",
        "Thai": r"^[\u0E00-\u0E7F]+$",
        "Vietnamese": r"^[A-Za-zàáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+$",
        # Indian languages
        "Hindi": r"^[\u0900-\u097F]+$",
        "Bengali": r"^[\u0980-\u09FF]+$",
        "Tamil": r"^[\u0B80-\u0BFF]+$",
        "Indian": r"^[\u0900-\u0DFF]+$",
        # Middle Eastern
        "Arabic": r"^[\u0600-\u06FF]+$",
        "Hebrew": r"^[\u0590-\u05FF]+$",
        # European languages
        "Russian": r"^[\u0400-\u04FF]+$",
        "Greek": r"^[\u0370-\u03FF\u1F00-\u1FFF]+$",
        "Armenian": r"^[\u0530-\u058F]+$",
        # Code and technical
        "Code": r"^[=.#@*&^_/\(\)\[\]\{\}\|\?\%\$<>+-:;,][A-Za-z0-9_]*$",
        "LaTeX": r"^\\[A-Za-z]+$",
        # Symbols and numbers
        "Numeric": r"^[0-9]+$",
        "Mathematical": r"^[\u2200-\u22FF\U0001D400-\U0001D7FF]+$",
        "Punctuation": r"^[\s\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u00A1-\u00BF\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F\uFF00-\uFFEF\u2190-\u21FF]+$",
        "Emoji": r"^[\U0001F300-\U0001FAFF\u2600-\u26FF]+$",
        # Latin-based languages (check last)
        "English": r'^[—–""' "'\"-]?[A-Za-z]+$",
        "French": r"^[A-Za-zàâäéèêëïîôùûüÿç]+$",
        "German": r"^[A-Za-zäöüßÄÖÜ]+$",
        "Spanish": r"^[A-Za-záéíóúñÑ]+$",
        "Portuguese": r"^[A-Za-zãõçéíóúâêîôû]+$",
        "Italian": r"^[A-Za-zàèéìíîòóùú]+$",
    }

    for lang, pattern in language_patterns.items():
        if re.match(pattern, token_stripped):
            return lang

    # Check for mixed content or other patterns
    if re.search(r"[A-Za-z]", token_stripped) and re.search(
        r"[0-9]", token_stripped
    ):
        return "Alphanumeric"

    return "Other"


def parse_token_category(token_name: str) -> str:
    """Parse token category - prioritizes custom tokens, then uses language detection."""
    token_name = token_name.strip()

    # Remove [ORIG] marker but still process the token
    is_original = token_name.startswith("[ORIG]")
    if is_original:
        token_name = token_name.replace("[ORIG]", "").strip()

    # Handle "ADDED_TOKEN: <token>" format from readable vocab
    if token_name.startswith("ADDED_TOKEN:"):
        # Extract the actual token name
        token_name = token_name.replace("ADDED_TOKEN:", "").strip()

    # Priority 1: Check for custom category tokens <a_*>, <b_*>, <c_*>, <d_*>
    if token_name.startswith("<") and token_name.endswith(">"):
        inner = token_name[1:-1].strip()

        # Check for main categories a, b, c, d
        if any(inner.startswith(f"{cat}_") for cat in "abcd"):
            category = inner[0]
            return f"Category {category.upper()}"

        # Check for special tokens
        if any(keyword in inner.lower() for keyword in ["end", "start", "pad"]):
            return "Special Token"

        # Other angle bracket tokens
        return "Special Token"

    # Priority 2: Use language detection for all other tokens
    language = detect_token_language(token_name)

    # If it was detected as "Added Token" but we couldn't categorize it further,
    # keep it as Added Token
    if language == "Added Token":
        return "Added Token"

    # Mark original tokens with a prefix for special handling if needed
    if is_original and language != "Other":
        return f"Original-{language}"

    return language


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba format."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_language_colors(use_rgba: bool = False) -> dict[str, str]:
    """Get color mapping for different languages/categories."""
    colors = {
        # Priority colors for custom categories (bright, distinct colors)
        "Category A": "#FF0000",  # Bright Red
        "Category B": "#00FF00",  # Bright Green
        "Category C": "#0080FF",  # Bright Blue
        "Category D": "#FFD700",  # Gold
        # Asian languages
        "Chinese": "#FF6B6B",
        "Japanese": "#4ECDC4",
        "Korean": "#45B7D1",
        "Thai": "#96CEB4",
        "Vietnamese": "#FD79A8",
        # Indian languages
        "Hindi": "#FDCB6E",
        "Bengali": "#6C5CE7",
        "Tamil": "#A29BFE",
        "Indian": "#FF7675",
        # Middle Eastern
        "Arabic": "#74B9FF",
        "Hebrew": "#A0E7E5",
        # European languages
        "English": "#81C784",
        "Russian": "#64B5F6",
        "Greek": "#BA68C8",
        "French": "#FFB74D",
        "German": "#FF8A65",
        "Spanish": "#F06292",
        "Portuguese": "#9575CD",
        "Italian": "#4FC3F7",
        # Technical
        "Code": "#455A64",
        "LaTeX": "#8D6E63",
        "Numeric": "#78909C",
        "Mathematical": "#5C6BC0",
        # Special tokens
        "Special Token": "#BA68C8",
        "Added Token": "#F48FB1",
        # Others
        "Punctuation": "#B0BEC5",
        "Emoji": "#FFD54F",
        "Whitespace": "#ECEFF1",
        "Other": "#9E9E9E",
        "Invalid UTF-8": "#D32F2F",
        "Alphanumeric": "#7E57C2",
    }

    # Add colors for original tokens (slightly faded versions)
    original_languages = [
        "Chinese",
        "Japanese",
        "Korean",
        "English",
        "Russian",
        "Arabic",
        "French",
        "German",
        "Spanish",
        "Italian",
        "Vietnamese",
        "Thai",
        "Hindi",
        "Bengali",
        "Tamil",
    ]

    for lang in original_languages:
        if lang in colors:
            if use_rgba:
                # Create rgba version with transparency for Plotly
                colors[f"Original-{lang}"] = hex_to_rgba(colors[lang], 0.6)
            else:
                # Keep hex for matplotlib
                colors[f"Original-{lang}"] = colors[lang] + "99"

    # Fallback for any Original-* not explicitly defined
    colors["Original-Other"] = "#90A4AE"

    return colors


def analyze_tokenizer_stats(tokenizer, model_path: str) -> dict:
    """Analyze tokenizer statistics."""
    stats = {
        "vocab_size": tokenizer.vocab_size,
        "added_tokens": len(tokenizer.added_tokens_encoder),
        "special_tokens": tokenizer.special_tokens_map,
        "vocab_length": len(tokenizer.vocab),
        "model_path": model_path,
    }

    print("\n=== Tokenizer Statistics ===")
    print(f"Model: {model_path}")
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"Added tokens: {stats['added_tokens']:,}")
    print(f"Special tokens: {stats['special_tokens']}")
    print(f"Total vocab length: {stats['vocab_length']:,}")

    return stats


def load_safetensor_embeddings(
    safetensor_path: str, layer_name: str = "model.embed_tokens.weight"
) -> torch.Tensor:
    """Load embeddings from a safetensor file."""
    from safetensors.torch import safe_open

    with safe_open(safetensor_path, framework="pt") as f:
        # Check available keys in the file
        available_keys = list(f.keys())

        if layer_name not in available_keys:
            # Try to find embedding layer automatically
            possible_names = [
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "model.model.language_model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "embed_tokens.weight",
                "model.model.embed_tokens.weight",
                "transformer.wte.weight",
                "transformer.embed_tokens.weight",
                "embeddings.weight",
                "shared.weight",
                "lm_head.weight",  # Sometimes embeddings are tied with lm_head
                "model.embed_tokens",  # Without .weight suffix
                "embed_tokens",
            ]

            # Also check for keys containing 'embed' or 'token'
            embed_keys = [
                k
                for k in available_keys
                if "embed" in k.lower() and "weight" in k.lower()
            ]

            found_key = None
            for possible_name in possible_names:
                if possible_name in available_keys:
                    found_key = possible_name
                    print(f"Found embedding layer at: {found_key}")
                    break

            if not found_key and embed_keys:
                found_key = embed_keys[0]
                print(f"Using first embedding-like layer found: {found_key}")

            if not found_key:
                print(f"Available keys in this file: {available_keys[:10]}...")
                return None

            layer_name = found_key

        embeddings = f.get_tensor(layer_name)

    if embeddings is not None and embeddings.dtype == torch.bfloat16:
        embeddings = embeddings.to(torch.float32)

    return embeddings


def merge_lora_embeddings(
    base_embeddings: torch.Tensor,
    lora_checkpoint: str,
    lora_alpha: float = None,
    lora_r: int = None,
) -> torch.Tensor:
    """Merge LoRA adapter weights with base embeddings if present."""
    import json
    
    # Check for adapter_config.json to get LoRA parameters
    adapter_config_path = os.path.join(lora_checkpoint, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
            if lora_alpha is None:
                lora_alpha = adapter_config.get("lora_alpha", 16)
            if lora_r is None:
                lora_r = adapter_config.get("r", 8)
            print(f"Found LoRA config: alpha={lora_alpha}, r={lora_r}")
    
    # Check for adapter weights
    adapter_path = os.path.join(lora_checkpoint, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(lora_checkpoint, "adapter_model.bin")
    
    if os.path.exists(adapter_path):
        print(f"Found LoRA adapter weights at: {adapter_path}")
        
        if adapter_path.endswith(".safetensors"):
            from safetensors.torch import safe_open
            with safe_open(adapter_path, framework="pt") as f:
                adapter_keys = list(f.keys())
                
                # Look for embed_tokens LoRA weights
                lora_A_key = None
                lora_B_key = None
                
                for key in adapter_keys:
                    if "embed_tokens" in key and "lora_A" in key:
                        lora_A_key = key
                    elif "embed_tokens" in key and "lora_B" in key:
                        lora_B_key = key
                
                if lora_A_key and lora_B_key:
                    print(f"Merging LoRA weights from {lora_A_key} and {lora_B_key}")
                    lora_A = f.get_tensor(lora_A_key)
                    lora_B = f.get_tensor(lora_B_key)
                    
                    # Convert to float32 if needed
                    if lora_A.dtype == torch.bfloat16:
                        lora_A = lora_A.to(torch.float32)
                    if lora_B.dtype == torch.bfloat16:
                        lora_B = lora_B.to(torch.float32)
                    
                    # Merge: W' = W + (B @ A) * (alpha / r)
                    scaling = lora_alpha / lora_r if lora_r != 0 else 1.0
                    lora_weights = (lora_B @ lora_A) * scaling
                    
                    # Add to base embeddings
                    merged_embeddings = base_embeddings.clone()
                    merged_embeddings += lora_weights.T  # Transpose because LoRA is (hidden_dim, r) @ (r, vocab_size)
                    
                    print(f"Successfully merged LoRA adapters (scaling={scaling:.2f})")
                    return merged_embeddings
                else:
                    print("No embed_tokens LoRA weights found in adapter")
        else:
            # Handle .bin format
            adapters = torch.load(adapter_path, map_location="cpu")
            
            lora_A_key = None
            lora_B_key = None
            
            for key in adapters.keys():
                if "embed_tokens" in key and "lora_A" in key:
                    lora_A_key = key
                elif "embed_tokens" in key and "lora_B" in key:
                    lora_B_key = key
            
            if lora_A_key and lora_B_key:
                print(f"Merging LoRA weights from {lora_A_key} and {lora_B_key}")
                lora_A = adapters[lora_A_key]
                lora_B = adapters[lora_B_key]
                
                # Convert to float32 if needed
                if lora_A.dtype == torch.bfloat16:
                    lora_A = lora_A.to(torch.float32)
                if lora_B.dtype == torch.bfloat16:
                    lora_B = lora_B.to(torch.float32)
                
                # Merge: W' = W + (B @ A) * (alpha / r)
                scaling = lora_alpha / lora_r if lora_r != 0 else 1.0
                lora_weights = (lora_B @ lora_A) * scaling
                
                # Add to base embeddings
                merged_embeddings = base_embeddings.clone()
                merged_embeddings += lora_weights.T
                
                print(f"Successfully merged LoRA adapters (scaling={scaling:.2f})")
                return merged_embeddings
            else:
                print("No embed_tokens LoRA weights found in adapter")
    else:
        print("No LoRA adapter files found, using base embeddings")
    
    return base_embeddings


def load_embeddings_from_model(
    model_path: str,
    layer_name: str = "model.embed_tokens.weight",
    sample_original_tokens: int = 1000,
    analyze_languages: bool = True,
    filter_languages: list[str] = None,
    seed: int = 42,
    lora_checkpoint: str = None,
) -> EmbeddingInfo:
    """Load embeddings from model with language analysis.
    
    Args:
        model_path: Path to base model
        layer_name: Name of the embedding layer
        sample_original_tokens: Number of original tokens to sample
        analyze_languages: Whether to analyze token languages
        filter_languages: Languages to filter for
        seed: Random seed
        lora_checkpoint: Path to LoRA checkpoint (if using LoRA)
    """
    embeddings = None

    # Check for safetensor files
    safetensor_files = sorted(
        [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    )

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    if len(safetensor_files) == 1:
        # Single file case
        safetensor_path = os.path.join(model_path, safetensor_files[0])
        print(f"Loading embeddings from: {safetensor_path}")
        embeddings = load_safetensor_embeddings(safetensor_path, layer_name)
    else:
        # Multiple files case - try each file until we find the embeddings
        print("Found multiple safetensor files:")
        print(safetensor_files)

        for file in safetensor_files:
            safetensor_path = os.path.join(model_path, file)
            print(f"Loading embeddings from: {safetensor_path}")
            try:
                embeddings = load_safetensor_embeddings(
                    safetensor_path, layer_name
                )
                if embeddings is not None:
                    print(f"Successfully loaded embeddings from {file}")
                    break
                print("Failed")
            except Exception as e:
                print(f"Failed: {str(e)[:100]}")
                continue

    if embeddings is None:
        raise RuntimeError(
            f"Could not load embedding layer '{layer_name}' from any safetensor file. "
            f"Try specifying a different layer name with --layer_name"
        )

    # Load tokenizer (from LoRA checkpoint if provided, otherwise from base model)
    tokenizer_path = lora_checkpoint if lora_checkpoint else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    
    # Check if we need to resize embeddings for new tokens
    if embeddings.shape[0] < vocab_size:
        print(f"\nResizing embeddings from {embeddings.shape[0]} to {vocab_size} tokens")
        # Create new embedding matrix with space for new tokens
        new_embeddings = torch.zeros((vocab_size, embeddings.shape[1]), dtype=embeddings.dtype)
        # Copy existing embeddings
        new_embeddings[:embeddings.shape[0]] = embeddings
        
        # If LoRA checkpoint exists, load new token embeddings from it
        if lora_checkpoint:
            # Try to load new token embeddings from checkpoint
            embed_tokens_path = os.path.join(lora_checkpoint, "adapter_model.safetensors")
            if not os.path.exists(embed_tokens_path):
                embed_tokens_path = os.path.join(lora_checkpoint, "adapter_model.bin")
            
            if os.path.exists(embed_tokens_path):
                if embed_tokens_path.endswith(".safetensors"):
                    from safetensors.torch import safe_open
                    with safe_open(embed_tokens_path, framework="pt") as f:
                        # Look for new token embeddings (usually stored as base_model.model.embed_tokens.modules_to_save.default.weight)
                        for key in f.keys():
                            if "embed_tokens" in key and "modules_to_save" in key:
                                print(f"Loading new token embeddings from {key}")
                                saved_embeddings = f.get_tensor(key)
                                if saved_embeddings.dtype == torch.bfloat16:
                                    saved_embeddings = saved_embeddings.to(torch.float32)
                                # Copy the new token embeddings
                                if saved_embeddings.shape[0] == vocab_size:
                                    new_embeddings = saved_embeddings
                                    print(f"Loaded complete embedding matrix with new tokens")
                                break
                else:
                    adapters = torch.load(embed_tokens_path, map_location="cpu")
                    for key in adapters.keys():
                        if "embed_tokens" in key and "modules_to_save" in key:
                            print(f"Loading new token embeddings from {key}")
                            saved_embeddings = adapters[key]
                            if saved_embeddings.dtype == torch.bfloat16:
                                saved_embeddings = saved_embeddings.to(torch.float32)
                            # Copy the new token embeddings
                            if saved_embeddings.shape[0] == vocab_size:
                                new_embeddings = saved_embeddings
                                print(f"Loaded complete embedding matrix with new tokens")
                            break
        
        embeddings = new_embeddings
    
    # Merge LoRA weights if checkpoint provided
    if lora_checkpoint:
        print(f"\nMerging LoRA checkpoint from: {lora_checkpoint}")
        embeddings = merge_lora_embeddings(embeddings, lora_checkpoint)

    # Analyze tokenizer
    stats = analyze_tokenizer_stats(tokenizer, model_path)

    # Get readable vocabulary if analyzing languages
    readable_vocab = {}
    if analyze_languages:
        print("\nConverting vocabulary to readable format...")
        readable_vocab = convert_to_readable_vocab(tokenizer, verbose=True)

    # Get new tokens (added tokens)
    new_token_ids = list(tokenizer.added_tokens_decoder.keys())

    # Filter for specific categories if they exist
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

    # Sample original tokens with optional language filtering
    original_token_ids = list(range(min(vocab_size, embeddings.shape[0])))
    original_token_ids = [
        tid for tid in original_token_ids if tid not in new_token_ids
    ]

    # Apply language filter if specified
    if filter_languages and analyze_languages:
        print(
            f"Filtering original tokens for languages: {', '.join(filter_languages)}"
        )
        filtered_token_ids = []

        for tid in tqdm(
            original_token_ids, desc="Filtering tokens by language"
        ):
            try:
                if tid in readable_vocab:
                    token_text = readable_vocab[tid]
                else:
                    token_text = tokenizer.decode(tid)

                # Remove special prefixes for language detection
                clean_text = (
                    token_text.replace("ADDED_TOKEN:", "")
                    .replace("INVALID UTF-8:", "")
                    .strip()
                )

                # Detect language
                lang = detect_token_language(clean_text)

                # Check if token matches any of the filter languages
                if lang in filter_languages:
                    filtered_token_ids.append(tid)

            except Exception:
                continue

        original_token_ids = filtered_token_ids
        print(
            f"Found {len(original_token_ids)} tokens matching filter languages"
        )

    random.seed(seed)
    num_samples = min(sample_original_tokens, len(original_token_ids))
    if num_samples > 0 and len(original_token_ids) > 0:
        sampled_original_ids = random.sample(original_token_ids, num_samples)
    else:
        sampled_original_ids = (
            original_token_ids[:num_samples] if original_token_ids else []
        )

    print(f"Sampled {len(sampled_original_ids)} original tokens")

    # Combine token IDs
    all_token_ids = new_token_ids + sampled_original_ids

    # Get token names and detect languages
    token_names = []
    token_languages = []

    for token_id in tqdm(all_token_ids, desc="Processing tokens"):
        try:
            if token_id in readable_vocab:
                name = readable_vocab[token_id]
            else:
                name = tokenizer.decode(token_id)

            # Mark original tokens
            if token_id in sampled_original_ids:
                name = f"[ORIG] {name}"

            token_names.append(name)

            # Detect language/category using parse_token_category
            if analyze_languages:
                # Use parse_token_category instead of detect_token_language
                # to properly handle custom categories
                category = parse_token_category(name)
                token_languages.append(category)

        except Exception:
            token_names.append(f"<token_{token_id}>")
            token_languages.append("Other")

    # Print language/category distribution
    if analyze_languages:
        print("\n=== Token Category & Language Distribution ===")
        lang_counts = {}
        for lang in token_languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Separate custom categories from languages
        custom_categories = [
            "Category A",
            "Category B",
            "Category C",
            "Category D",
        ]

        # Print custom categories first
        print("\nCustom Categories:")
        for cat in custom_categories:
            if cat in lang_counts:
                count = lang_counts[cat]
                print(
                    f"  {cat}: {count} tokens ({count / len(token_languages) * 100:.1f}%)"
                )

        # Print other categories
        print("\nLanguages & Other Categories:")
        sorted_langs = sorted(
            [
                (lang, count)
                for lang, count in lang_counts.items()
                if lang not in custom_categories
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for lang, count in sorted_langs[:20]:  # Show top 20
            print(
                f"  {lang}: {count} tokens ({count / len(token_languages) * 100:.1f}%)"
            )

    # Extract embeddings (filter out invalid token IDs)
    valid_token_ids = [tid for tid in all_token_ids if tid < embeddings.shape[0]]
    if len(valid_token_ids) < len(all_token_ids):
        print(f"Warning: Filtered out {len(all_token_ids) - len(valid_token_ids)} invalid token IDs")
        # Update token names and languages accordingly
        valid_indices = [i for i, tid in enumerate(all_token_ids) if tid < embeddings.shape[0]]
        token_names = [token_names[i] for i in valid_indices]
        if token_languages:
            token_languages = [token_languages[i] for i in valid_indices]
        all_token_ids = valid_token_ids
    
    selected_embeddings = embeddings[all_token_ids, :].numpy()

    return EmbeddingInfo(
        selected_embeddings, token_names, all_token_ids, token_languages
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
                n_iter=args.tsne_iterations,
                metric="cosine" if args.use_cosine else "euclidean",
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
                n_iter=args.tsne_iterations,
                metric="cosine" if args.use_cosine else "euclidean",
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
    token_languages: list[str],
    title: str,
    output_path: str,
    args: argparse.Namespace,
):
    """Create static matplotlib visualization with combined category/language coloring."""
    plt.figure(figsize=(args.fig_width, args.fig_height))

    # Get colors based on combined category/language detection (matplotlib supports hex with alpha)
    color_map = get_language_colors(use_rgba=False)

    # Always use parse_token_category which handles both custom tokens and languages
    categories = [parse_token_category(name) for name in token_names]

    colors = [color_map.get(cat, "#808080") for cat in categories]

    # Create scatter plot
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        alpha=args.point_alpha,
        s=args.point_size,
        c=colors,
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

    # Create legend (prioritize Category A-D, then languages)
    unique_categories = list(set(categories))

    # Separate custom categories from languages
    custom_categories = ["Category A", "Category B", "Category C", "Category D"]
    existing_custom = [
        cat for cat in custom_categories if cat in unique_categories
    ]
    other_categories = [
        cat for cat in unique_categories if cat not in custom_categories
    ]

    # Build legend elements with custom categories first
    legend_elements = []

    # Add custom categories first (they'll appear at top of legend)
    for category in existing_custom:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[category],
                markersize=12,
                label=category,
                markeredgewidth=2,
                markeredgecolor="black",
            )  # Add edge for emphasis
        )

    # Add other categories (languages) sorted alphabetically
    for category in sorted(other_categories)[:16]:  # Limit total to ~20
        if category in color_map:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map[category],
                    markersize=10,
                    label=category,
                )
            )

    if legend_elements:
        plt.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
            ncol=2 if len(legend_elements) > 10 else 1,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to: {output_path}")


def create_interactive_visualization(
    embeddings_2d: np.ndarray,
    token_names: list[str],
    token_languages: list[str],
    title: str,
    output_path: str,
    args: argparse.Namespace,
):
    """Create interactive plotly visualization with combined category/language coloring."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return

    fig = go.Figure()

    # Get colors (use rgba for Plotly compatibility)
    color_map = get_language_colors(use_rgba=True)

    # Always use parse_token_category which handles both custom tokens and languages
    categories = [parse_token_category(name) for name in token_names]

    # Group tokens by category
    category_groups = {}
    for i, (name, cat) in enumerate(zip(token_names, categories, strict=False)):
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(i)

    # Add traces for each category
    for category, indices in sorted(category_groups.items()):
        color = color_map.get(category, "#808080")

        fig.add_trace(
            go.Scatter(
                x=[embeddings_2d[i, 0] for i in indices],
                y=[embeddings_2d[i, 1] for i in indices],
                mode="markers",
                name=category,
                marker=dict(size=8, color=color, opacity=0.7),
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
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
    )

    fig.write_html(output_path)
    print(f"Saved interactive visualization to: {output_path}")


def compute_token_similarities(
    embeddings: np.ndarray, token_ids: list[int], tokenizer, top_k: int = 10
) -> dict:
    """Compute and return top-k similar tokens for each token."""
    print(f"\nComputing top-{top_k} similar tokens...")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = embeddings / norms

    # Compute similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)

    similarities = {}
    for i, token_id in enumerate(token_ids[:100]):  # Limit to first 100 tokens
        # Get top-k similar tokens (excluding self)
        sim_scores = similarity_matrix[i]
        sim_scores[i] = -1  # Exclude self

        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        top_scores = sim_scores[top_indices]

        similar_tokens = []
        for idx, score in zip(top_indices, top_scores, strict=False):
            try:
                similar_token = tokenizer.decode(token_ids[idx])
            except:
                similar_token = f"<token_{token_ids[idx]}>"
            similar_tokens.append((similar_token, float(score)))

        try:
            token_name = tokenizer.decode(token_id)
        except:
            token_name = f"<token_{token_id}>"

        similarities[token_name] = similar_tokens

    return similarities


def save_analysis_results(
    output_dir: str,
    tokenizer_stats: dict,
    language_distribution: dict,
    token_similarities: dict = None,
):
    """Save analysis results to JSON files."""
    # Save tokenizer statistics
    stats_path = os.path.join(output_dir, "tokenizer_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_stats, f, indent=2, ensure_ascii=False)
    print(f"Saved tokenizer statistics to: {stats_path}")

    # Save language distribution
    lang_path = os.path.join(output_dir, "language_distribution.json")
    with open(lang_path, "w", encoding="utf-8") as f:
        json.dump(language_distribution, f, indent=2, ensure_ascii=False)
    print(f"Saved language distribution to: {lang_path}")

    # Save token similarities if computed
    if token_similarities:
        sim_path = os.path.join(output_dir, "token_similarities.json")
        with open(sim_path, "w", encoding="utf-8") as f:
            json.dump(token_similarities, f, indent=2, ensure_ascii=False)
        print(f"Saved token similarities to: {sim_path}")


def list_model_layers(model_path: str):
    """List all available layers in the model."""
    safetensor_files = sorted(
        [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    )

    if not safetensor_files:
        print("No safetensor files found")
        return

    from safetensors.torch import safe_open

    all_keys = set()
    for file in safetensor_files:
        safetensor_path = os.path.join(model_path, file)
        with safe_open(safetensor_path, framework="pt") as f:
            keys = list(f.keys())
            all_keys.update(keys)
            print(f"\n{file}: {len(keys)} layers")

    # Filter for potential embedding layers
    embed_keys = sorted([k for k in all_keys if "embed" in k.lower()])

    print("\n=== Potential Embedding Layers ===")
    for key in embed_keys:
        print(f"  {key}")

    print("\n=== All Layers (first 50) ===")
    for i, key in enumerate(sorted(all_keys)[:50]):
        print(f"  {key}")

    if len(all_keys) > 50:
        print(f"  ... and {len(all_keys) - 50} more layers")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced token embedding visualization with language analysis"
    )

    # Input/Output arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Base model path (for LoRA) or full model path (for non-LoRA)",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory containing adapter weights and tokenizer",
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
        default="auto",
        help="Layer name for safetensor loading (use 'auto' for automatic detection)",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="List all available layers in the model and exit",
    )

    # Sampling arguments
    parser.add_argument(
        "--sample_original",
        type=int,
        default=1000,
        help="Number of original tokens to sample (default: 1000)",
    )
    parser.add_argument(
        "--filter_languages",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "Russian",
            "Arabic",
            "French",
            "German",
            "Spanish",
            "Italian",
            "Portuguese",
            "Thai",
            "Vietnamese",
            "Hindi",
            "Code",
            "Numeric",
        ],
        help="Filter original tokens to only include specified languages (e.g., --filter_languages Chinese English)",
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
    parser.add_argument(
        "--tsne_iterations",
        type=int,
        default=400,
        help="Number of t-SNE iterations",
    )
    parser.add_argument(
        "--use_cosine",
        action="store_true",
        help="Use cosine distance for t-SNE",
    )

    # Visualization appearance arguments
    parser.add_argument(
        "--fig_width", type=int, default=20, help="Figure width in inches"
    )
    parser.add_argument(
        "--fig_height", type=int, default=16, help="Figure height in inches"
    )
    parser.add_argument(
        "--point_size", type=int, default=50, help="Scatter plot point size"
    )
    parser.add_argument(
        "--point_alpha",
        type=float,
        default=0.6,
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
        default=1400,
        help="Interactive plot width in pixels",
    )
    parser.add_argument(
        "--interactive_height",
        type=int,
        default=900,
        help="Interactive plot height in pixels",
    )

    # Analysis arguments
    parser.add_argument(
        "--skip_analysis", action="store_true", help="Skip language analysis"
    )
    parser.add_argument(
        "--compute_similarities",
        action="store_true",
        help="Compute token similarities",
    )
    parser.add_argument(
        "--top_k_similar",
        type=int,
        default=10,
        help="Number of similar tokens to compute",
    )

    args = parser.parse_args()

    # List layers if requested
    if args.list_layers:
        list_model_layers(args.model_path)
        return

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.model_path
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings with language analysis
    print(f"Loading embeddings from: {args.model_path}")

    # Use automatic detection if layer_name is 'auto'
    layer_name = None if args.layer_name == "auto" else args.layer_name

    embedding_info = load_embeddings_from_model(
        args.model_path,
        layer_name if layer_name else "model.embed_tokens.weight",
        args.sample_original,
        not args.skip_analysis,
        args.filter_languages,
        args.seed,
        lora_checkpoint=args.lora_checkpoint,
    )

    embeddings = embedding_info.embeddings
    token_names = embedding_info.token_names
    token_ids = embedding_info.token_ids
    token_languages = embedding_info.token_languages

    # Apply max_tokens limit if specified
    if args.max_tokens and len(token_names) > args.max_tokens:
        print(f"Limiting tokens from {len(token_names)} to {args.max_tokens}")
        random.seed(args.seed)
        indices = random.sample(range(len(token_names)), args.max_tokens)
        embeddings = embeddings[indices]
        token_names = [token_names[i] for i in indices]
        token_ids = [token_ids[i] for i in indices]
        if token_languages:
            token_languages = [token_languages[i] for i in indices]

    print(
        f"Processing {len(token_names)} tokens with {embeddings.shape[1]}-dimensional embeddings"
    )

    # Compute language distribution
    language_distribution = {}
    if token_languages:
        for lang in token_languages:
            language_distribution[lang] = language_distribution.get(lang, 0) + 1

    # Compute token similarities if requested
    token_similarities = None
    if args.compute_similarities:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        token_similarities = compute_token_similarities(
            embeddings, token_ids, tokenizer, args.top_k_similar
        )

    # Save analysis results
    if not args.skip_analysis:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        stats = analyze_tokenizer_stats(tokenizer, args.model_path)
        save_analysis_results(
            args.output_dir, stats, language_distribution, token_similarities
        )

    # Generate visualizations
    methods = ["tsne", "pca"] if args.method == "both" else [args.method]

    for method in methods:
        print(f"\n=== {method.upper()} Visualization ===")

        # Perform dimension reduction
        embeddings_2d, title = perform_dimension_reduction(
            embeddings, method, args
        )

        # Add color mode to title
        title += " (Categories A-D + Languages)"

        # Create static visualization
        output_path = os.path.join(args.output_dir, f"embeddings_{method}.png")
        create_static_visualization(
            embeddings_2d,
            token_names,
            token_languages,
            title,
            output_path,
            args,
        )

        # Create interactive visualization if requested
        if args.interactive:
            interactive_path = os.path.join(
                args.output_dir, f"embeddings_{method}_interactive.html"
            )
            create_interactive_visualization(
                embeddings_2d,
                token_names,
                token_languages,
                title,
                interactive_path,
                args,
            )

    print(f"\n✓ All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
