"""
Sparse Attention Visualizer — Main Experiment Runner
=====================================================
Trains transformer classifiers with different attention mechanisms,
evaluates them, and generates visualizations.

Usage:
    python main.py                           # Quick demo with synthetic data
    python main.py --dataset imdb --samples 500 --epochs 3
    python main.py --k-values 5 10 20 50 100 # Sweep k values
"""

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_model
from utils import (
    SimpleTokenizer,
    TextDataset,
    evaluate,
    get_single_example_attention,
    load_agnews,
    load_imdb,
    train_one_epoch,
)
from attention_viz import (
    generate_all_visualizations,
    plot_metrics_comparison,
    plot_speed_accuracy_tradeoff,
)


def run_experiment(
    attention_type: str,
    k: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    n_classes: int,
    epochs: int,
    lr: float,
    max_seq_len: int,
    d_model: int,
) -> Dict:
    """Train and evaluate a single model configuration."""
    label = {
        "full": "Full Attention",
        "random": f"Random Sparse (k={k})",
        "topk": f"Top-k Sparse (k={k})",
    }.get(attention_type, attention_type)
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"{'='*60}")

    model = build_model(
        vocab_size=vocab_size,
        attention_type=attention_type,
        k=k,
        n_classes=n_classes,
        d_model=d_model,
        max_seq_len=max_seq_len,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss:.4f}")

    # Evaluate
    metrics = evaluate(model, test_loader)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  Time:     {metrics['inference_time_ms']:.1f} ms")
    print(f"  Memory:   {metrics['peak_memory_mb']:.2f} MB")

    return {
        "model": model,
        "metrics": metrics,
        "attention_type": attention_type,
        "k": k,
        "param_count": param_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Sparse Attention Visualizer")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["imdb", "agnews", "synthetic"])
    parser.add_argument("--samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=15000)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--example-text",
        type=str,
        default="This movie was absolutely fantastic, I loved every minute of the brilliant acting and storyline",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Sparse Attention Visualizer")
    print("  MiniMax-Style Top-k on CPU")
    print("=" * 60)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Samples:    {args.samples}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  k values:   {args.k_values}")
    print(f"  Seq length: {args.max_seq_len}")
    print(f"  d_model:    {args.d_model}")
    print("=" * 60)

    # ----- Load data -----
    print("\n[1/5] Loading data...")
    if args.dataset == "imdb":
        train_texts, train_labels, test_texts, test_labels = load_imdb(args.samples)
        n_classes = 2
    elif args.dataset == "agnews":
        train_texts, train_labels, test_texts, test_labels = load_agnews(args.samples)
        n_classes = 4
    else:
        from utils import _synthetic_sentiment
        train_texts, train_labels, test_texts, test_labels = _synthetic_sentiment(args.samples)
        n_classes = 2

    print(f"  Train: {len(train_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")
    print(f"  Classes: {n_classes}")

    # ----- Tokenize -----
    print("\n[2/5] Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    actual_vocab = tokenizer.vocab_len()
    print(f"  Vocabulary: {actual_vocab} tokens")

    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_seq_len)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ----- Run experiments -----
    print("\n[3/5] Running experiments...")

    all_results = {}
    metrics_by_k = {}

    # Full attention baseline
    full_result = run_experiment(
        "full", 0, train_loader, test_loader, actual_vocab, n_classes,
        args.epochs, args.lr, args.max_seq_len, args.d_model,
    )
    all_results["full"] = full_result
    baseline_metrics = full_result["metrics"]

    # Random sparse (use median k)
    median_k = sorted(args.k_values)[len(args.k_values) // 2]
    random_result = run_experiment(
        "random", median_k, train_loader, test_loader, actual_vocab, n_classes,
        args.epochs, args.lr, args.max_seq_len, args.d_model,
    )
    all_results["random"] = random_result

    # Top-k for each k value
    for k in args.k_values:
        topk_result = run_experiment(
            "topk", k, train_loader, test_loader, actual_vocab, n_classes,
            args.epochs, args.lr, args.max_seq_len, args.d_model,
        )
        metrics_by_k[k] = topk_result["metrics"]
        if k == median_k:
            all_results["topk"] = topk_result

    # ----- Generate visualizations -----
    print("\n[4/5] Generating visualizations...")

    # Get attention maps for example text
    example_results = {}
    for name, res in all_results.items():
        example_data = get_single_example_attention(
            res["model"], args.example_text, tokenizer, max_len=64,
        )
        example_results[name] = example_data
    tokens = example_results["full"]["tokens"]

    generate_all_visualizations(
        example_results, tokens,
        metrics_by_k=metrics_by_k,
        baseline_metrics=baseline_metrics,
        output_dir=args.output,
    )

    # Metrics comparison chart
    comparison_metrics = {}
    for name, res in all_results.items():
        comparison_metrics[name] = res["metrics"]
    plot_metrics_comparison(
        comparison_metrics,
        save_path=os.path.join(args.output, "metrics_comparison.png"),
    )

    # ----- Save metrics JSON -----
    print("\n[5/5] Saving metrics...")
    metrics_dir = os.path.join(args.output, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "samples": args.samples,
        "epochs": args.epochs,
        "k_values": args.k_values,
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "f1": baseline_metrics["f1"],
            "inference_time_ms": baseline_metrics["inference_time_ms"],
            "peak_memory_mb": baseline_metrics["peak_memory_mb"],
        },
        "topk_results": {
            str(k): {
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "inference_time_ms": m["inference_time_ms"],
                "peak_memory_mb": m["peak_memory_mb"],
                "speedup": baseline_metrics["inference_time_ms"] / max(m["inference_time_ms"], 0.001),
            }
            for k, m in metrics_by_k.items()
        },
    }

    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {metrics_dir}/summary.json")

    # ----- Final summary -----
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  {'Type':<25} {'Accuracy':>10} {'F1':>10} {'Time(ms)':>10} {'Speedup':>10}")
    print("  " + "-" * 65)
    print(f"  {'Full Attention':<25} {baseline_metrics['accuracy']:>10.4f} {baseline_metrics['f1']:>10.4f} {baseline_metrics['inference_time_ms']:>10.1f} {'1.00x':>10}")

    for k in sorted(metrics_by_k.keys()):
        m = metrics_by_k[k]
        speedup = baseline_metrics["inference_time_ms"] / max(m["inference_time_ms"], 0.001)
        print(f"  {'Top-k (k=' + str(k) + ')':<25} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['inference_time_ms']:>10.1f} {speedup:>9.2f}x")

    print(f"\n  Output directory: {os.path.abspath(args.output)}")
    print("=" * 60)
    print("  Done! Check the results/ folder for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
