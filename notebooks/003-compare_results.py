#!/usr/bin/env python3
"""
Compare results between Baseline and LoRA training
"""

import json
import glob
import numpy as np
from pathlib import Path


def load_latest_results():
    """Load the most recent baseline and LoRA results"""
    baseline_files = glob.glob("results_baseline_*.json")
    lora_files = glob.glob("results_lora_*.json")

    if not baseline_files:
        print("❌ No baseline results found! Run 001-baseline-llama.py first")
        return None, None

    if not lora_files:
        print("❌ No LoRA results found! Run 002-lora-llama.py first")
        return None, None

    # Get the most recent files
    latest_baseline = max(baseline_files, key=lambda x: Path(x).stat().st_mtime)
    latest_lora = max(lora_files, key=lambda x: Path(x).stat().st_mtime)

    print(f"📊 Loading results:")
    print(f"  Baseline: {latest_baseline}")
    print(f"  LoRA: {latest_lora}")

    with open(latest_baseline, "r") as f:
        baseline_results = json.load(f)

    with open(latest_lora, "r") as f:
        lora_results = json.load(f)

    return baseline_results, lora_results


def compare_results(baseline, lora):
    """Compare baseline vs LoRA results"""

    print("\n" + "=" * 80)
    print("🔥 LLAMA 3.2 1B SENTIMENT ANALYSIS: BASELINE vs LoRA COMPARISON")
    print("=" * 80)

    # Performance comparison
    print("\n📈 PERFORMANCE METRICS")
    print("-" * 50)

    metrics = ["accuracy", "f1_score", "precision", "recall", "loss"]

    for metric in metrics:
        baseline_val = baseline[metric]
        lora_val = lora[metric]

        if metric == "loss":
            # For loss, lower is better
            diff = baseline_val - lora_val
            improvement = f"{diff:+.4f}"
            status = "📉" if diff > 0 else "📈"
        else:
            # For other metrics, higher is better
            diff = lora_val - baseline_val
            improvement = f"{diff:+.4f}"
            status = "📈" if diff > 0 else "📉"

        print(
            f"  {metric.title():12} | Baseline: {baseline_val:.4f} | LoRA: {lora_val:.4f} | Δ: {improvement} {status}"
        )

    # Parameter efficiency
    print("\n💪 PARAMETER EFFICIENCY")
    print("-" * 50)

    baseline_total = baseline["total_params"]
    baseline_trainable = baseline["trainable_params"]
    lora_total = lora["total_params"]
    lora_trainable = lora["trainable_params"]

    param_reduction = (1 - lora_trainable / baseline_trainable) * 100
    efficiency_ratio = baseline_trainable / lora_trainable

    print(f"  Total Parameters:")
    print(f"    Baseline: {baseline_total:,}")
    print(f"    LoRA:     {lora_total:,}")

    print(f"\n  Trainable Parameters:")
    print(
        f"    Baseline: {baseline_trainable:,} ({100 * baseline_trainable / baseline_total:.1f}%)"
    )
    print(
        f"    LoRA:     {lora_trainable:,} ({100 * lora_trainable / lora_total:.1f}%)"
    )

    print(f"\n  🎯 Efficiency Gains:")
    print(f"    Parameter Reduction: {param_reduction:.1f}%")
    print(f"    Efficiency Ratio: {efficiency_ratio:.1f}x fewer trainable parameters")

    # Performance per parameter
    baseline_acc_per_param = baseline["accuracy"] / baseline_trainable * 1e6
    lora_acc_per_param = lora["accuracy"] / lora_trainable * 1e6

    print(f"    Accuracy per Million Params:")
    print(f"      Baseline: {baseline_acc_per_param:.2f}")
    print(
        f"      LoRA:     {lora_acc_per_param:.2f} ({lora_acc_per_param / baseline_acc_per_param:.1f}x better)"
    )

    # Per-class analysis
    print("\n🎭 PER-CLASS PERFORMANCE")
    print("-" * 50)

    classes = ["Negative", "Positive"]
    for class_name in classes:
        if class_name in baseline["classification_report"]:
            b_metrics = baseline["classification_report"][class_name]
            l_metrics = lora["classification_report"][class_name]

            print(f"  {class_name}:")
            print(
                f"    Precision | Baseline: {b_metrics['precision']:.3f} | LoRA: {l_metrics['precision']:.3f} | Δ: {l_metrics['precision'] - b_metrics['precision']:+.3f}"
            )
            print(
                f"    Recall    | Baseline: {b_metrics['recall']:.3f} | LoRA: {l_metrics['recall']:.3f} | Δ: {l_metrics['recall'] - b_metrics['recall']:+.3f}"
            )
            print(
                f"    F1-Score  | Baseline: {b_metrics['f1-score']:.3f} | LoRA: {l_metrics['f1-score']:.3f} | Δ: {l_metrics['f1-score'] - b_metrics['f1-score']:+.3f}"
            )

    # Confusion matrices
    print("\n🔍 CONFUSION MATRICES")
    print("-" * 50)

    def print_confusion_matrix(cm, title):
        print(f"  {title}:")
        print(f"    Predicted:  Neg   Pos")
        print(f"    Actual Neg: {cm[0][0]:3d}   {cm[0][1]:3d}")
        print(f"    Actual Pos: {cm[1][0]:3d}   {cm[1][1]:3d}")

        # Calculate accuracy from confusion matrix
        accuracy = (cm[0][0] + cm[1][1]) / sum(sum(row) for row in cm)
        print(f"    Accuracy: {accuracy:.3f}")

    print_confusion_matrix(baseline["confusion_matrix"], "Baseline")
    print()
    print_confusion_matrix(lora["confusion_matrix"], "LoRA")

    # Overall assessment
    print("\n🏆 OVERALL ASSESSMENT")
    print("-" * 50)

    acc_diff = lora["accuracy"] - baseline["accuracy"]
    f1_diff = lora["f1_score"] - baseline["f1_score"]

    print(f"  LoRA vs Baseline:")

    if param_reduction > 90:
        efficiency_grade = "🔥 EXCELLENT"
    elif param_reduction > 80:
        efficiency_grade = "⭐ GREAT"
    elif param_reduction > 50:
        efficiency_grade = "👍 GOOD"
    else:
        efficiency_grade = "📊 MODERATE"

    print(
        f"    Parameter Efficiency: {efficiency_grade} ({param_reduction:.1f}% reduction)"
    )

    if acc_diff > 0.01:
        performance_grade = "🚀 SUPERIOR"
    elif acc_diff > -0.01:
        performance_grade = "🎯 COMPARABLE"
    elif acc_diff > -0.05:
        performance_grade = "⚠️ SLIGHTLY LOWER"
    else:
        performance_grade = "❌ SIGNIFICANTLY LOWER"

    print(f"    Performance: {performance_grade} ({acc_diff:+.3f} accuracy)")

    # Final verdict
    if param_reduction > 80 and acc_diff > -0.02:
        verdict = "🎉 LoRA is highly effective! Major parameter savings with minimal performance impact."
    elif param_reduction > 50 and acc_diff > -0.05:
        verdict = (
            "✅ LoRA is effective! Good parameter savings with acceptable performance."
        )
    elif param_reduction > 90:
        verdict = "⚖️ LoRA trades performance for efficiency. Consider if the trade-off is acceptable."
    else:
        verdict = "🤔 LoRA benefits unclear. May need hyperparameter tuning."

    print(f"\n  📝 Verdict: {verdict}")

    print("\n" + "=" * 80)


def main():
    """Main comparison function"""
    baseline_results, lora_results = load_latest_results()

    if baseline_results is None or lora_results is None:
        print("\n❌ Cannot compare - missing results files")
        print("Please run both notebooks first:")
        print("  1. 001-baseline-llama.py")
        print("  2. 002-lora-llama.py")
        return

    compare_results(baseline_results, lora_results)


if __name__ == "__main__":
    main()
