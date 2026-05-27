"""
Attention Visualization
========================
Generates publication-quality visualizations:
- Attention heatmaps (per-head, averaged)
- Token selection maps (which tokens get attended to)
- Side-by-side comparison (full vs sparse)
- Speed-accuracy tradeoff curves
- Interactive Plotly charts
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from models import build_model
from utils import SimpleTokenizer, get_single_example_attention


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLORS = {
    "full": "#2196F3",
    "random": "#FF9800",
    "topk": "#4CAF50",
    "bg": "#1a1a2e",
    "text": "#e0e0e0",
    "grid": "#333355",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Attention Heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: List[str],
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    head_idx: int = 0,
    layer_idx: int = 0,
    max_tokens: int = 30,
):
    """
    Plot a single attention head's weights as a heatmap.

    Args:
        attention: (n_heads, seq_len, seq_len) or (seq_len, seq_len)
        tokens: list of token strings
        head_idx: which head to plot
        layer_idx: for title labeling
        max_tokens: truncate display for readability
    """
    if attention.ndim == 3:
        attn = attention[head_idx]
    else:
        attn = attention

    n = min(len(tokens), max_tokens, attn.shape[0])
    attn = attn[:n, :n]
    display_tokens = tokens[:n]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        cmap="YlOrRd",
        vmin=0,
        ax=ax,
        square=True,
        cbar_kws={"label": "Attention Weight", "shrink": 0.8},
    )
    ax.set_title(f"{title}\nLayer {layer_idx}, Head {head_idx}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Key (attended to)", fontsize=11)
    ax.set_ylabel("Query (attending from)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_attention_comparison(
    results: Dict[str, Dict],
    tokens: List[str],
    save_path: Optional[str] = None,
    head_idx: int = 0,
    max_tokens: int = 25,
):
    """
    Side-by-side heatmaps for full vs random vs topk attention.
    """
    n_types = len(results)
    fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 6))
    if n_types == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        attn = data["attention_weights"][0]  # layer 0
        if attn.ndim == 3:
            attn = attn[head_idx]
        n = min(len(tokens), max_tokens, attn.shape[0])
        attn = attn[:n, :n]
        display_tokens = tokens[:n]

        sns.heatmap(
            attn,
            xticklabels=display_tokens,
            yticklabels=display_tokens,
            cmap="YlOrRd",
            vmin=0,
            ax=ax,
            square=True,
        )
        label = {"full": "Full Attention", "random": "Random Sparse", "topk": "Top-k Sparse"}
        ax.set_title(label.get(name, name), fontsize=13, fontweight="bold")
        ax.set_xlabel("Key", fontsize=10)
        ax.set_ylabel("Query", fontsize=10)
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)

    plt.suptitle("Attention Pattern Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Token selection map
# ---------------------------------------------------------------------------

def plot_token_selection(
    attention: np.ndarray,
    tokens: List[str],
    query_idx: int = 0,
    title: str = "Token Selection",
    save_path: Optional[str] = None,
    head_idx: int = 0,
    max_tokens: int = 40,
):
    """
    Bar chart showing attention weights from a specific query token to all keys.
    Highlights which tokens are being attended to (selected) vs ignored.
    """
    if attention.ndim == 3:
        attn_row = attention[head_idx, query_idx]
    else:
        attn_row = attention[query_idx]

    n = min(len(tokens), max_tokens, len(attn_row))
    attn_row = attn_row[:n]
    display_tokens = tokens[:n]

    # Color: green if attended (> threshold), gray if ignored
    threshold = 1.0 / n  # uniform baseline
    colors = ["#4CAF50" if w > threshold else "#cccccc" for w in attn_row]

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(range(n), attn_row, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Attention Weight", fontsize=11)
    ax.set_title(
        f'{title}\nQuery token: "{display_tokens[query_idx]}"',
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label="Uniform baseline")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Sparsity pattern visualization
# ---------------------------------------------------------------------------

def plot_sparsity_pattern(
    attention: np.ndarray,
    tokens: List[str],
    title: str = "Sparsity Pattern",
    save_path: Optional[str] = None,
    head_idx: int = 0,
    max_tokens: int = 30,
):
    """
    Binary plot: white = attended, black = ignored.
    Shows the structural sparsity pattern clearly.
    """
    if attention.ndim == 3:
        attn = attention[head_idx]
    else:
        attn = attention

    n = min(len(tokens), max_tokens, attn.shape[0])
    attn = attn[:n, :n]
    display_tokens = tokens[:n]

    # Binary: 1 if weight > small epsilon
    binary = (attn > 1e-6).astype(float)
    sparsity = 1.0 - binary.mean()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(binary, cmap="Greys", aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(display_tokens, fontsize=7)
    ax.set_xlabel("Key", fontsize=10)
    ax.set_ylabel("Query", fontsize=10)
    ax.set_title(
        f"{title}\nSparsity: {sparsity:.1%} of connections pruned",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Speed-accuracy tradeoff
# ---------------------------------------------------------------------------

def plot_speed_accuracy_tradeoff(
    metrics_by_k: Dict[int, Dict],
    baseline_metrics: Dict,
    save_path: Optional[str] = None,
):
    """
    Scatter plot: x = speedup ratio, y = accuracy.
    Each point is a different k value.
    """
    ks = sorted(metrics_by_k.keys())
    speedups = [
        baseline_metrics["inference_time_ms"] / metrics_by_k[k]["inference_time_ms"]
        for k in ks
    ]
    accuracies = [metrics_by_k[k]["accuracy"] * 100 for k in ks]
    baseline_acc = baseline_metrics["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline
    ax.axhline(y=baseline_acc, color=COLORS["full"], linestyle="--", alpha=0.7, label="Full Attention")
    ax.axvline(x=1.0, color=COLORS["full"], linestyle="--", alpha=0.3)

    # Top-k points
    scatter = ax.scatter(speedups, accuracies, c=[COLORS["topk"]] * len(ks), s=120, zorder=5, edgecolors="white")
    for k, sp, ac in zip(ks, speedups, accuracies):
        ax.annotate(
            f"k={k}",
            (sp, ac),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Speedup Ratio (vs Full Attention)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Speed vs Accuracy Tradeoff\nTop-k Sparse Attention", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Metrics comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    all_metrics: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart comparing accuracy, F1, time, memory across attention types.
    """
    names = list(all_metrics.keys())
    labels = {
        "full": "Full Attention",
        "random": "Random Sparse",
        "topk": "Top-k Sparse",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_keys = [
        ("accuracy", "Accuracy", "#4CAF50"),
        ("f1", "F1 Score", "#2196F3"),
        ("inference_time_ms", "Inference Time (ms)", "#FF9800"),
        ("peak_memory_mb", "Peak Memory (MB)", "#E91E63"),
    ]

    for ax, (key, ylabel, color) in zip(axes, metric_keys):
        values = [all_metrics[n][key] for n in names]
        display_names = [labels.get(n, n) for n in names]
        bars = ax.bar(display_names, values, color=color, edgecolor="white", alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, values):
            fmt = f"{val:.3f}" if val < 10 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Attention Mechanism Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Interactive Plotly heatmap
# ---------------------------------------------------------------------------

def plot_interactive_heatmap(
    attention: np.ndarray,
    tokens: List[str],
    title: str = "Interactive Attention Heatmap",
    save_path: Optional[str] = None,
    head_idx: int = 0,
    max_tokens: int = 40,
):
    """Generate an interactive Plotly heatmap saved as HTML."""
    if not HAS_PLOTLY:
        print("  [SKIP] Plotly not installed, skipping interactive heatmap.")
        return

    if attention.ndim == 3:
        attn = attention[head_idx]
    else:
        attn = attention

    n = min(len(tokens), max_tokens, attn.shape[0])
    attn = attn[:n, :n]
    display_tokens = tokens[:n]

    fig = go.Figure(
        data=go.Heatmap(
            z=attn,
            x=display_tokens,
            y=display_tokens,
            colorscale="YlOrRd",
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Key (attended to)",
        yaxis_title="Query (attending from)",
        width=700,
        height=650,
    )
    fig.update_yaxes(autorange="reversed")

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")


def plot_interactive_tradeoff(
    metrics_by_k: Dict[int, Dict],
    baseline_metrics: Dict,
    save_path: Optional[str] = None,
):
    """Interactive speed-accuracy tradeoff with Plotly."""
    if not HAS_PLOTLY:
        print("  [SKIP] Plotly not installed.")
        return

    ks = sorted(metrics_by_k.keys())
    speedups = [
        baseline_metrics["inference_time_ms"] / metrics_by_k[k]["inference_time_ms"]
        for k in ks
    ]
    accuracies = [metrics_by_k[k]["accuracy"] * 100 for k in ks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=speedups,
        y=accuracies,
        mode="markers+text",
        text=[f"k={k}" for k in ks],
        textposition="top right",
        marker=dict(size=14, color=COLORS["topk"], line=dict(color="white", width=2)),
        name="Top-k Sparse",
    ))
    fig.add_hline(
        y=baseline_metrics["accuracy"] * 100,
        line_dash="dash",
        line_color=COLORS["full"],
        annotation_text="Full Attention baseline",
    )
    fig.update_layout(
        title="Speed vs Accuracy Tradeoff",
        xaxis_title="Speedup Ratio",
        yaxis_title="Accuracy (%)",
        width=700,
        height=500,
    )

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.write_html(save_path)
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 7. All-in-one visualization runner
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    results: Dict[str, Dict],
    tokens: List[str],
    metrics_by_k: Optional[Dict[int, Dict]] = None,
    baseline_metrics: Optional[Dict] = None,
    output_dir: str = "results",
):
    """
    Generate the full set of visualizations.

    Args:
        results: dict of {attention_type: {attention_weights, ...}}
        tokens: list of token strings for the example sentence
        metrics_by_k: optional dict {k: eval_metrics} for tradeoff curve
        baseline_metrics: optional full-attention eval metrics
        output_dir: where to save everything
    """
    print("\n=== Generating Visualizations ===\n")

    heatmap_dir = os.path.join(output_dir, "heatmaps")
    examples_dir = os.path.join(output_dir, "examples")
    interactive_dir = os.path.join(output_dir, "interactive")

    # Per-type visualizations
    for name, data in results.items():
        attn = data["attention_weights"][0]  # layer 0
        label = {"full": "Full Attention", "random": "Random Sparse", "topk": "Top-k Sparse"}.get(name, name)

        plot_attention_heatmap(
            attn, tokens,
            title=label,
            save_path=os.path.join(heatmap_dir, f"heatmap_{name}.png"),
        )

        plot_sparsity_pattern(
            attn, tokens,
            title=f"{label} Sparsity",
            save_path=os.path.join(heatmap_dir, f"sparsity_{name}.png"),
        )

        plot_token_selection(
            attn, tokens,
            query_idx=0,
            title=f"{label} — Token Selection",
            save_path=os.path.join(examples_dir, f"token_selection_{name}.png"),
        )

        plot_interactive_heatmap(
            attn, tokens,
            title=f"{label} (Interactive)",
            save_path=os.path.join(interactive_dir, f"heatmap_{name}.html"),
        )

    # Comparison
    if len(results) > 1:
        plot_attention_comparison(
            results, tokens,
            save_path=os.path.join(heatmap_dir, "comparison.png"),
        )

    # Tradeoff curves
    if metrics_by_k and baseline_metrics:
        plot_speed_accuracy_tradeoff(
            metrics_by_k, baseline_metrics,
            save_path=os.path.join(output_dir, "speed_accuracy_tradeoff.png"),
        )
        plot_interactive_tradeoff(
            metrics_by_k, baseline_metrics,
            save_path=os.path.join(interactive_dir, "tradeoff.html"),
        )

    print("\n=== All visualizations saved ===\n")


# ---------------------------------------------------------------------------
# CLI entry point: quick single-sentence visualization
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns on a single sentence")
    parser.add_argument(
        "--example",
        type=str,
        default="I love this movie, it's absolutely fantastic and wonderful!",
        help="Text to visualize",
    )
    parser.add_argument("--k", type=int, default=10, help="k for sparse attention")
    parser.add_argument("--output", type=str, default="results/examples", help="Output directory")
    args = parser.parse_args()

    print(f'Text: "{args.example}"')
    print(f"k = {args.k}\n")

    # Build a small tokenizer from the example itself (demo mode)
    tokenizer = SimpleTokenizer(vocab_size=500)
    tokenizer.build_vocab([args.example])

    vocab_size = tokenizer.vocab_len()

    for attn_type in ["full", "random", "topk"]:
        print(f"--- {attn_type.upper()} ---")
        model = build_model(vocab_size, attention_type=attn_type, k=args.k, max_seq_len=128)
        result = get_single_example_attention(model, args.example, tokenizer, max_len=64)
        tokens = result["tokens"]
        attn = result["attention_weights"][0]

        plot_attention_heatmap(
            attn, tokens,
            title=f"{attn_type.upper()} Attention",
            save_path=os.path.join(args.output, f"heatmap_{attn_type}.png"),
        )
        plot_token_selection(
            attn, tokens,
            query_idx=0,
            title=f"{attn_type.upper()} — Token Selection",
            save_path=os.path.join(args.output, f"selection_{attn_type}.png"),
        )
        plot_sparsity_pattern(
            attn, tokens,
            title=f"{attn_type.upper()} Sparsity",
            save_path=os.path.join(args.output, f"sparsity_{attn_type}.png"),
        )

    print("\nDone! Check", args.output)


if __name__ == "__main__":
    main()
