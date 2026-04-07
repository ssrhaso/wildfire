"""Cross-model comparison of freezing ablation results.

Produces a scatter plot of test accuracy vs percentage of trainable
parameters across all three architectures, highlighting the efficiency
of layer freezing for transfer learning.

Usage:
    python src/compare_models.py
    python src/compare_models.py --results-dir results --output-dir results/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyse_results import CONFIG_LABELS

# Publication-quality matplotlib configuration
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

MODEL_STYLE = {
    "vit": {"colour": "#D55E00", "marker": "o", "label": "ViT-B/16"},
    "resnet": {"colour": "#0072B2", "marker": "s", "label": "ResNet-50"},
    "hybrid": {"colour": "#009E73", "marker": "^", "label": "Hybrid CNN-ViT"},
}

# BatchNorm-frozen variants are excluded to avoid cluttering the plot
EXCLUDE_SUFFIX = "_bnfrozen"


def load_model_results(results_dir: Path, model_name: str) -> List[Dict]:
    """Load per-config aggregated results for a model from JSON files."""
    model_dir = results_dir / model_name
    if not model_dir.exists():
        return []

    configs = []
    for config_dir in sorted(model_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        if config_name.endswith(EXCLUDE_SUFFIX):
            continue

        accs, trainable_pcts = [], []
        for json_file in config_dir.glob("seed_*.json"):
            with open(json_file) as f:
                data = json.load(f)
            accs.append(data["test_acc"])
            # Recompute exact percentage from raw counts (avoids rounding to 0.0)
            total = data.get("num_total_params", 0)
            trainable = data.get("num_trainable_params", 0)
            pct = (trainable / total * 100) if total > 0 else data.get("trainable_pct", 0)
            trainable_pcts.append(pct)

        if accs:
            configs.append({
                "config": config_name,
                "label": CONFIG_LABELS.get(config_name, config_name),
                "mean_acc": np.mean(accs),
                "std_acc": np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
                "trainable_pct": np.mean(trainable_pcts),
                "n_seeds": len(accs),
            })

    return configs


def plot_accuracy_vs_params(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Scatter plot of test accuracy vs trainable parameter percentage.

    Each point is one freeze configuration (averaged across seeds).
    Error bars show +/-1 std across seeds.  The x-axis uses symmetric
    log scale so that both sub-1% and near-100% values are visible.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for model_name, configs in all_results.items():
        if not configs:
            continue

        style = MODEL_STYLE[model_name]
        pcts = np.array([c["trainable_pct"] for c in configs])
        accs = np.array([c["mean_acc"] * 100 for c in configs])
        stds = np.array([c["std_acc"] * 100 for c in configs])

        ax.errorbar(
            pcts,
            accs,
            yerr=stds,
            fmt=style["marker"],
            color=style["colour"],
            label=style["label"],
            markersize=7,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            markeredgewidth=0.6,
            markeredgecolor="white",
            zorder=3,
        )

        # Annotate the best config for each model
        best = max(configs, key=lambda c: c["mean_acc"])
        ax.annotate(
            best["label"],
            xy=(best["trainable_pct"], best["mean_acc"] * 100),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=7,
            color=style["colour"],
            fontstyle="italic",
        )

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Trainable Parameters (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. Parameter Efficiency")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"cross_model_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved: cross_model_comparison.png/pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-model comparison plot")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <results-dir>/analysis).",
    )
    args = p.parse_args()

    plt.rcParams.update(PAPER_RC)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Loading results across all models\n")

    all_results = {}
    for model_name in ("vit", "resnet", "hybrid"):
        configs = load_model_results(results_dir, model_name)
        all_results[model_name] = configs
        n_configs = len(configs)
        n_total = sum(c["n_seeds"] for c in configs)
        print(f"  {model_name:<8} {n_configs} configs, {n_total} runs")

    total_configs = sum(len(v) for v in all_results.values())
    if total_configs == 0:
        print("\n  No results found. Run experiments first.")
        sys.exit(1)

    plot_accuracy_vs_params(all_results, output_dir)

    # Print summary table
    print(f"\n  {'Model':<12} {'Config':<35} {'Acc (%)':<16} {'Trainable %':>12}")
    print(f"  {'-' * 12} {'-' * 35} {'-' * 16} {'-' * 12}")
    for model_name, configs in all_results.items():
        for c in sorted(configs, key=lambda x: x["mean_acc"], reverse=True):
            acc_str = f"{c['mean_acc'] * 100:.2f} +/- {c['std_acc'] * 100:.2f}"
            print(
                f"  {model_name:<12} {c['label']:<35} {acc_str:<16} {c['trainable_pct']:>11.3f}"
            )

    print(f"\n  All outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()
