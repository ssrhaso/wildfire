"""Statistical analysis of freezing ablation results.

Reads JSON results and produces summary tables, statistical tests,
and plots matching the methodology from Liu & Ahmad (2026, ICISS).

Usage:
    python src/analyse_results.py --model vit --results-dir results
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


CONFIG_ORDER = {
    "vit": [
        "freeze_none",
        "freeze_patch",
        "freeze_patch_blocks0-3",
        "freeze_patch_blocks0-5",
        "freeze_patch_blocks0-8",
        "freeze_patch_blocks0-11",
    ],
    "resnet": [
        "freeze_none",
        "freeze_conv1",
        "freeze_conv1_layer1",
        "freeze_conv1_layer1-2",
        "freeze_conv1_layer1-3",
        "freeze_conv1_layer1-4",
    ],
    "hybrid": [
        "freeze_none",
        "freeze_CNN",
        "freeze_CNN_proj",
        "freeze_transformer_only",
        "freeze_transformer_proj",
        "freeze_CNN_proj_transformer",
        "freeze_CNN_proj_blocks0-3",
        "freeze_CNN_proj_blocks0-5",
        "freeze_CNN_proj_blocks0-8",
        "freeze_CNN_proj_blocks0-11",
        "freeze_blocks0-3",
        "freeze_blocks0-5",
        "freeze_blocks0-8",
        "freeze_blocks0-11",
        "freeze_none_bnfrozen",
        "freeze_transformer_only_bnfrozen",
        "freeze_transformer_proj_bnfrozen",
        "freeze_blocks0-3_bnfrozen",
        "freeze_blocks0-5_bnfrozen",
        "freeze_blocks0-8_bnfrozen",
        "freeze_blocks0-11_bnfrozen",
    ],
}

CONFIG_LABELS = {
    # ViT
    "freeze_none": "Freeze none",
    "freeze_patch": "Freeze patch",
    "freeze_patch_blocks0-3": "Freeze patch+0-3",
    "freeze_patch_blocks0-5": "Freeze patch+0-5",
    "freeze_patch_blocks0-8": "Freeze patch+0-8",
    "freeze_patch_blocks0-11": "Freeze patch+0-11",
    # ResNet
    "freeze_conv1": "Freeze conv1",
    "freeze_conv1_layer1": "Freeze conv1+L1",
    "freeze_conv1_layer1-2": "Freeze conv1+L1-2",
    "freeze_conv1_layer1-3": "Freeze conv1+L1-3",
    "freeze_conv1_layer1-4": "Freeze conv1+L1-4",
    # Hybrid
    "freeze_CNN": "Freeze CNN",
    "freeze_CNN_proj": "Freeze CNN+proj",
    "freeze_transformer_only": "Freeze transformer",
    "freeze_transformer_proj": "Freeze transformer+proj",
    "freeze_CNN_proj_transformer": "Freeze CNN+proj+transformer",
    "freeze_CNN_proj_blocks0-3": "Freeze CNN+proj+blocks 0-3",
    "freeze_CNN_proj_blocks0-5": "Freeze CNN+proj+blocks 0-5",
    "freeze_CNN_proj_blocks0-8": "Freeze CNN+proj+blocks 0-8",
    "freeze_CNN_proj_blocks0-11": "Freeze CNN+proj+blocks 0-11",
    "freeze_blocks0-3": "Freeze blocks 0-3",
    "freeze_blocks0-5": "Freeze blocks 0-5",
    "freeze_blocks0-8": "Freeze blocks 0-8",
    "freeze_blocks0-11": "Freeze blocks 0-11",
    "freeze_none_bnfrozen": "Freeze none (BN frozen)",
    "freeze_transformer_only_bnfrozen": "Freeze transformer (BN frozen)",
    "freeze_transformer_proj_bnfrozen": "Freeze transformer+proj (BN frozen)",
    "freeze_blocks0-3_bnfrozen": "Freeze blocks 0-3 (BN frozen)",
    "freeze_blocks0-5_bnfrozen": "Freeze blocks 0-5 (BN frozen)",
    "freeze_blocks0-8_bnfrozen": "Freeze blocks 0-8 (BN frozen)",
    "freeze_blocks0-11_bnfrozen": "Freeze blocks 0-11 (BN frozen)",
}


def load_results(results_dir: Path, model: str) -> Dict[str, List[Dict]]:
    """Load all JSON results for a model, grouped by freeze config."""
    model_dir = results_dir / model
    if not model_dir.exists():
        print(f"  No results directory: {model_dir}")
        sys.exit(1)

    grouped = {}
    for config_dir in sorted(model_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        runs = []
        for json_file in sorted(config_dir.glob("seed_*.json")):
            with open(json_file) as f:
                runs.append(json.load(f))
        if runs:
            grouped[config_name] = runs
    return grouped


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def summary_table(grouped: Dict[str, List[Dict]], config_order: List[str]) -> pd.DataFrame:
    """Build summary table: config, mean accuracy, std, sorted by mean desc."""
    rows = []
    for config in config_order:
        if config not in grouped:
            continue
        accs = np.array([r["test_acc"] for r in grouped[config]])
        f1_fires = np.array([r.get("test_f1_fire", float("nan")) for r in grouped[config]])
        f1_macros = np.array([
            r.get("best_val_f1_macro", float("nan")) for r in grouped[config]
        ])
        rows.append({
            "config": config,
            "label": CONFIG_LABELS.get(config, config),
            "mean_acc": accs.mean(),
            "std_acc": accs.std(ddof=1) if len(accs) > 1 else 0.0,
            "mean_f1_fire": float(np.nanmean(f1_fires)),
            "std_f1_fire": float(np.nanstd(f1_fires, ddof=1)) if len(f1_fires) > 1 else 0.0,
            "n_seeds": len(accs),
            "mean_trainable_pct": np.mean([r["trainable_pct"] for r in grouped[config]]),
        })
    df = pd.DataFrame(rows).sort_values("mean_acc", ascending=False).reset_index(drop=True)
    return df


def statistical_tests(
    grouped: Dict[str, List[Dict]], config_order: List[str]
) -> pd.DataFrame:
    """Welch's t-test and Cohen's d for all config pairs."""
    configs_with_data = [c for c in config_order if c in grouped]
    rows = []

    for c1, c2 in combinations(configs_with_data, 2):
        a1 = np.array([r["test_acc"] for r in grouped[c1]])
        a2 = np.array([r["test_acc"] for r in grouped[c2]])

        if len(a1) < 2 or len(a2) < 2:
            continue

        t_stat, p_val = ttest_ind(a1, a2, equal_var=False)
        d = cohens_d(a1, a2)

        rows.append({
            "config_1": CONFIG_LABELS.get(c1, c1),
            "config_2": CONFIG_LABELS.get(c2, c2),
            "mean_1": a1.mean(),
            "mean_2": a2.mean(),
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": d,
            "significant": p_val < 0.05,
        })

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return df


def plot_boxplot(
    grouped: Dict[str, List[Dict]],
    config_order: List[str],
    output_dir: Path,
    model: str,
) -> None:
    """Box plot of test accuracy distribution per config."""
    data, labels = [], []
    for config in config_order:
        if config not in grouped:
            continue
        accs = [r["test_acc"] for r in grouped[config]]
        data.extend(accs)
        labels.extend([CONFIG_LABELS.get(config, config)] * len(accs))

    df = pd.DataFrame({"Accuracy": data, "Config": labels})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Config", y="Accuracy", ax=ax, palette="Set2")
    ax.set_title(f"Test accuracy distribution - {model.upper()} freezing configs")
    ax.set_xlabel("")
    ax.set_ylabel("Test Accuracy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"boxplot_accuracy.{fmt}", dpi=300)
    plt.close(fig)


def plot_val_curves(
    grouped: Dict[str, List[Dict]],
    config_order: List[str],
    output_dir: Path,
    model: str,
) -> None:
    """Average validation accuracy across seeds, one line per config."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    for i, config in enumerate(config_order):
        if config not in grouped:
            continue
        runs = grouped[config]
        max_epochs = max(len(r["train_history"]) for r in runs)

        epoch_accs = []
        for ep in range(max_epochs):
            accs = [
                r["train_history"][ep]["val_acc"]
                for r in runs
                if ep < len(r["train_history"])
            ]
            epoch_accs.append(np.mean(accs) if accs else np.nan)

        ax.plot(
            range(1, max_epochs + 1),
            epoch_accs,
            label=CONFIG_LABELS.get(config, config),
            color=cmap(i),
            linewidth=1.5,
        )

    ax.set_title(f"Validation accuracy comparison - {model.upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"val_curves.{fmt}", dpi=300)
    plt.close(fig)


def plot_train_val_curves(
    grouped: Dict[str, List[Dict]],
    config_order: List[str],
    output_dir: Path,
    model: str,
) -> None:
    """Train+val curves for best and worst config (matching ICISS Figs 3-6)."""
    summary = []
    for config in config_order:
        if config not in grouped:
            continue
        accs = np.array([r["test_acc"] for r in grouped[config]])
        summary.append((config, accs.mean()))

    if len(summary) < 2:
        return

    summary.sort(key=lambda x: x[1], reverse=True)
    best_config = summary[0][0]
    worst_config = summary[-1][0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, config, title_prefix in [
        (axes[0], best_config, "Best"),
        (axes[1], worst_config, "Worst"),
    ]:
        runs = grouped[config]
        max_epochs = max(len(r["train_history"]) for r in runs)

        train_accs, val_accs = [], []
        for ep in range(max_epochs):
            ta = [r["train_history"][ep]["train_acc"] for r in runs if ep < len(r["train_history"])]
            va = [r["train_history"][ep]["val_acc"] for r in runs if ep < len(r["train_history"])]
            train_accs.append(np.mean(ta) if ta else np.nan)
            val_accs.append(np.mean(va) if va else np.nan)

        epochs = range(1, max_epochs + 1)
        ax.plot(epochs, train_accs, label="Training Accuracy", color="#2196F3")
        ax.plot(epochs, val_accs, label="Validation Accuracy", color="#FF9800")
        ax.set_title(f"{title_prefix}: {CONFIG_LABELS.get(config, config)}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Training dynamics - {model.upper()}", fontsize=13)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"train_val_curves.{fmt}", dpi=300)
    plt.close(fig)


def plot_f1_curves(
    grouped: Dict[str, List[Dict]],
    config_order: List[str],
    output_dir: Path,
    model: str,
) -> None:
    """Average validation F1 (fire class) across seeds, one line per config."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    for i, config in enumerate(config_order):
        if config not in grouped:
            continue
        runs = grouped[config]
        max_epochs = max(len(r["train_history"]) for r in runs)

        epoch_f1s = []
        for ep in range(max_epochs):
            f1s = [
                r["train_history"][ep]["val_f1_fire"]
                for r in runs
                if ep < len(r["train_history"]) and "val_f1_fire" in r["train_history"][ep]
            ]
            epoch_f1s.append(np.mean(f1s) if f1s else np.nan)

        if all(np.isnan(v) for v in epoch_f1s):
            continue

        ax.plot(
            range(1, max_epochs + 1),
            epoch_f1s,
            label=CONFIG_LABELS.get(config, config),
            color=cmap(i),
            linewidth=1.5,
        )

    ax.set_title(f"Validation F1 (fire class) comparison - {model.upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score (fire)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"val_f1_fire_curves.{fmt}", dpi=300)
    plt.close(fig)


def plot_lr_schedule(
    grouped: Dict[str, List[Dict]],
    config_order: List[str],
    output_dir: Path,
    model: str,
) -> None:
    """Learning rate schedule per freeze config (first seed of each)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    for i, config in enumerate(config_order):
        if config not in grouped:
            continue
        run = grouped[config][0]
        epochs = [h["epoch"] for h in run["train_history"]]
        lrs = [h["lr"] for h in run["train_history"]]

        ax.plot(
            epochs,
            lrs,
            label=CONFIG_LABELS.get(config, config),
            color=cmap(i),
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    ax.set_title(f"Learning rate schedule per freeze config - {model.upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"lr_schedule.{fmt}", dpi=300)
    plt.close(fig)


def plot_confusion_matrices(
    grouped: Dict[str, List[Dict]],
    config_order: List[str],
    output_dir: Path,
    model: str,
) -> None:
    """Side-by-side confusion matrices for best vs worst config."""
    summary = []
    for config in config_order:
        if config not in grouped:
            continue
        accs = np.array([r["test_acc"] for r in grouped[config]])
        summary.append((config, accs.mean()))

    if len(summary) < 2:
        return

    summary.sort(key=lambda x: x[1], reverse=True)
    best_config = summary[0][0]
    worst_config = summary[-1][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, config, title_prefix in [
        (axes[0], best_config, "Best"),
        (axes[1], worst_config, "Worst"),
    ]:
        runs = grouped[config]
        cms = [np.array(r["test_confusion_matrix"]) for r in runs if "test_confusion_matrix" in r]
        if not cms:
            continue
        avg_cm = np.mean(cms, axis=0).astype(int)

        sns.heatmap(
            avg_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Nofire", "Fire"],
            yticklabels=["Nofire", "Fire"],
            ax=ax,
        )
        ax.set_title(f"{title_prefix}: {CONFIG_LABELS.get(config, config)}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle(f"Confusion matrices - {model.upper()}", fontsize=13)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"confusion_matrices.{fmt}", dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse freezing ablation results")
    p.add_argument("--model", type=str, default="vit", choices=["vit", "resnet", "hybrid"])
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "analysis" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading results from {results_dir / args.model}\n")
    grouped = load_results(results_dir, args.model)

    if not grouped:
        print("  No results found.")
        sys.exit(1)

    config_order = CONFIG_ORDER.get(args.model, sorted(grouped.keys()))

    print("  Summary Table\n")
    summary_df = summary_table(grouped, config_order)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n  Saved: {output_dir / 'summary.csv'}\n")

    print("  Statistical Comparisons\n")
    stats_df = statistical_tests(grouped, config_order)
    sig_df = stats_df[stats_df["significant"]]

    if len(sig_df) > 0:
        print(sig_df[["config_1", "config_2", "p_value", "cohens_d"]].to_string(index=False))
    else:
        print("  No statistically significant differences found (p < 0.05)")

    stats_df.to_csv(output_dir / "statistics.csv", index=False)
    print(f"\n  Saved: {output_dir / 'statistics.csv'}\n")

    print("  Generating Plots\n")

    plot_boxplot(grouped, config_order, output_dir, args.model)
    print(f"  Saved: boxplot_accuracy.png/pdf")

    plot_val_curves(grouped, config_order, output_dir, args.model)
    print(f"  Saved: val_curves.png/pdf")

    plot_train_val_curves(grouped, config_order, output_dir, args.model)
    print(f"  Saved: train_val_curves.png/pdf")

    plot_f1_curves(grouped, config_order, output_dir, args.model)
    print(f"  Saved: val_f1_fire_curves.png/pdf")

    plot_lr_schedule(grouped, config_order, output_dir, args.model)
    print(f"  Saved: lr_schedule.png/pdf")

    plot_confusion_matrices(grouped, config_order, output_dir, args.model)
    print(f"  Saved: confusion_matrices.png/pdf")

    print(f"\n  All outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()

