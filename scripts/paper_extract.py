"""One-shot extraction of paper-ready stats from existing JSON results.

Reads results/{model}/{config}/seed_*.json and prints copy-paste-ready
numbers for the UWE paper. Does not retrain or re-evaluate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_runs(model: str, config: str) -> list[dict]:
    cdir = RESULTS / model / config
    if not cdir.exists():
        return []
    return [json.load(open(p)) for p in sorted(cdir.glob("seed_*.json"))]


def load_all(model: str) -> dict[str, list[dict]]:
    out = {}
    for cdir in sorted((RESULTS / model).iterdir()):
        if cdir.is_dir():
            runs = [json.load(open(p)) for p in sorted(cdir.glob("seed_*.json"))]
            if runs:
                out[cdir.name] = runs
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def welch_df(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    num = (s1 / n1 + s2 / n2) ** 2
    den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    return float(num / den)


def compare(model: str, c1: str, c2: str) -> dict:
    r1 = load_runs(model, c1)
    r2 = load_runs(model, c2)
    a1 = np.array([r["test_acc"] for r in r1])
    a2 = np.array([r["test_acc"] for r in r2])
    f1 = np.array([r.get("test_f1_fire", np.nan) for r in r1])
    f2 = np.array([r.get("test_f1_fire", np.nan) for r in r2])
    t, p = ttest_ind(a1, a2, equal_var=False)
    return {
        "c1": c1, "c2": c2,
        "n1": len(a1), "n2": len(a2),
        "m1": a1.mean(), "s1": a1.std(ddof=1),
        "m2": a2.mean(), "s2": a2.std(ddof=1),
        "f1m1": np.nanmean(f1), "f1m2": np.nanmean(f2),
        "delta_pp": (a1.mean() - a2.mean()) * 100,
        "t": float(t), "p": float(p),
        "d": cohens_d(a1, a2),
        "df": welch_df(a1, a2),
    }


def fmt_p(p: float) -> str:
    if p == 0:
        return "0"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.2g}"


def section1():
    print("\n" + "=" * 70)
    print("SECTION 1: ViT freeze_patch_blocks0-3 / blocks0-5")
    print("=" * 70)
    for cfg in ("freeze_patch_blocks0-3", "freeze_patch_blocks0-5"):
        runs = load_runs("vit", cfg)
        seeds = sorted([r["seed"] for r in runs])
        accs = np.array([r["test_acc"] for r in runs])
        f1s = np.array([r["test_f1_fire"] for r in runs])
        print(f"\n  {cfg}: seeds present = {seeds} (n={len(runs)})")
        print(f"    test_acc:    mean={accs.mean():.4f}  std={accs.std(ddof=1):.4f}  "
              f"individual={['%.4f'%a for a in accs]}")
        print(f"    test_f1_fire mean={f1s.mean():.4f}  std={f1s.std(ddof=1):.4f}  "
              f"individual={['%.4f'%f for f in f1s]}")
        # also print which seeds in {0,5,10,15,20} ran
        wanted = {0, 5, 10, 15, 20}
        missing = sorted(wanted - set(seeds))
        print(f"    missing seeds: {missing if missing else 'none'}")


def section2():
    print("\n" + "=" * 70)
    print("SECTION 2: Compute hours")
    print("=" * 70)
    total_secs = 0.0
    arch_secs = defaultdict(float)
    arch_runs = defaultdict(int)
    gpu_secs = defaultdict(float)
    gpu_runs = defaultdict(int)
    n_total = 0
    for model in ("vit", "resnet", "hybrid"):
        for cfg, runs in load_all(model).items():
            for r in runs:
                t = r.get("total_train_time_seconds", 0.0) or 0.0
                gpu = r.get("gpu_name", "unknown")
                total_secs += t
                arch_secs[model] += t
                arch_runs[model] += 1
                gpu_secs[gpu] += t
                gpu_runs[gpu] += 1
                n_total += 1
    print(f"\n  Total runs counted: {n_total}")
    print(f"  Total train time: {total_secs:.0f} s = {total_secs/3600:.2f} GPU-h")
    print("\n  By architecture:")
    for k in ("vit", "resnet", "hybrid"):
        print(f"    {k:8s}: {arch_runs[k]:3d} runs  {arch_secs[k]/3600:6.2f} GPU-h  "
              f"({arch_secs[k]:.0f} s)")
    print("\n  By GPU device:")
    for k, v in sorted(gpu_secs.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {gpu_runs[k]:3d} runs  {v/3600:6.2f} GPU-h")


def section3():
    print("\n" + "=" * 70)
    print("SECTION 3: Three headline pairs (Welch, Cohen's d, df)")
    print("=" * 70)
    cases = [
        ("vit", "freeze_patch", "freeze_patch_blocks0-8",
         "ViT freeze_patch vs freeze_patch_blocks0-8",
         {"delta": 2.91, "p": 3.2e-7, "d": -13.1}),
        ("resnet", "freeze_conv1_layer1-3", "freeze_conv1_layer1-4",
         "RN50 freeze_conv1_layer1-3 vs freeze_conv1_layer1-4",
         {"delta": 2.10, "p": 1.7e-5, "d": 6.40}),
        ("hybrid", "freeze_blocks0-8", "freeze_backbone_proj_blocks0-8",
         "Hybrid freeze_blocks0-8 vs freeze_backbone_proj_blocks0-8",
         {"delta": 6.29, "p": float("nan"), "d": 11.77}),
    ]
    for model, c1, c2, label, claim in cases:
        r = compare(model, c1, c2)
        print(f"\n  {label}")
        print(f"    {c1}: {r['m1']:.4f} +/- {r['s1']:.4f}  "
              f"{c2}: {r['m2']:.4f} +/- {r['s2']:.4f}")
        print(f"    Delta = {r['delta_pp']:+.4f} pp")
        print(f"    p = {r['p']:.3e}    Cohen's d = {r['d']:.2f}    df = {r['df']:.2f}")
        # 5% relative tolerance flag
        flags = []
        for key, ours, val in [
            ("delta", r["delta_pp"], claim["delta"]),
            ("p", r["p"], claim["p"]),
            ("d", r["d"], claim["d"]),
        ]:
            if val != val:  # NaN means skip
                continue
            denom = abs(val) if val != 0 else 1.0
            rel = abs(abs(ours) - abs(val)) / denom
            if rel > 0.05:
                flags.append(f"{key}: paper={val} ours={ours:.4f} (rel={rel:.1%})")
        if flags:
            print(f"    ** FLAG mismatch >5%: {flags}".encode("ascii", "replace").decode())
        else:
            print(f"    OK: matches claim within 5%")


def section4():
    print("\n" + "=" * 70)
    print("SECTION 4: Hybrid alternative headline pairs")
    print("=" * 70)
    pairs = [
        ("freeze_backbone", "freeze_blocks0-8_bnfrozen", "current paper headline"),
        ("freeze_backbone", "freeze_none", "best vs full FT"),
        ("freeze_blocks0-8", "freeze_backbone_proj_blocks0-8", "projection-layer effect"),
        ("freeze_backbone", "freeze_transformer_only", "which component carries transfer"),
    ]
    rows = []
    for c1, c2, label in pairs:
        r = compare("hybrid", c1, c2)
        rows.append((label, c1, c2, r))
        print(f"\n  {label}: {c1} vs {c2}")
        print(f"    means: {r['m1']:.4f} vs {r['m2']:.4f}   Δ = {r['delta_pp']:+.4f} pp")
        print(f"    p = {r['p']:.3e}   d = {r['d']:.2f}   df = {r['df']:.2f}")
    # ratio of |Δ| (practical signal) to log10(p) — and to |d|
    print("\n  Practical-signal ratio (Δpp / |d|, larger = more interpretable Δ per σ):")
    for label, c1, c2, r in rows:
        ratio = abs(r["delta_pp"]) / abs(r["d"]) if r["d"] != 0 else float("inf")
        print(f"    {label:40s}  Δ={r['delta_pp']:+.2f}pp  d={r['d']:+.2f}  "
              f"Δ/|d|={ratio:.4f}")


def section5():
    print("\n" + "=" * 70)
    print("SECTION 5: ResNet-50 plateau verification — all pairwise p<0.05")
    print("=" * 70)
    grouped = load_all("resnet")
    cfgs = list(grouped.keys())
    sig = []
    for c1, c2 in combinations(cfgs, 2):
        a1 = np.array([r["test_acc"] for r in grouped[c1]])
        a2 = np.array([r["test_acc"] for r in grouped[c2]])
        t, p = ttest_ind(a1, a2, equal_var=False)
        if p < 0.05:
            d = cohens_d(a1, a2)
            sig.append((c1, c2, (a1.mean() - a2.mean()) * 100, p, d))
    sig.sort(key=lambda x: x[3])
    print(f"\n  Significant pairs (n={len(sig)}/{len(list(combinations(cfgs,2)))}):")
    for c1, c2, dpp, p, d in sig:
        flag = "(involves L1-4)" if "layer1-4" in c1 or "layer1-4" in c2 else ""
        print(f"    {c1:25s} vs {c2:25s}  Δ={dpp:+.4f}pp  p={p:.3e}  d={d:+.2f} {flag}")
    # Verify the plateau claim
    all_l14 = all(("layer1-4" in c1) or ("layer1-4" in c2) for c1, c2, _, _, _ in sig)
    print(f"\n  Claim 'all 4 sig pairs involve freeze_conv1_layer1-4': {all_l14}")
    print(f"  Claim '4 sig comparisons total': {len(sig)==4}")


def section6():
    print("\n" + "=" * 70)
    print("SECTION 6: Macro metrics for best-of-architecture")
    print("=" * 70)
    best = [
        ("vit", "freeze_patch_blocks0-8"),
        ("resnet", "freeze_conv1_layer1-3"),
        ("hybrid", "freeze_backbone"),
    ]
    for model, cfg in best:
        runs = load_runs(model, cfg)
        # Macro F1: average of per-class F1
        macro_f1 = np.array([
            (r["test_f1_fire"] + r["test_f1_nofire"]) / 2 for r in runs
        ])
        prec = np.array([r["test_precision_fire"] for r in runs])
        rec = np.array([r["test_recall_fire"] for r in runs])
        f1f = np.array([r["test_f1_fire"] for r in runs])
        acc = np.array([r["test_acc"] for r in runs])
        print(f"\n  {model.upper()}  {cfg}  (n={len(runs)})")
        print(f"    test_acc        : {acc.mean():.4f} ± {acc.std(ddof=1):.4f}")
        print(f"    test_f1_fire    : {f1f.mean():.4f} ± {f1f.std(ddof=1):.4f}")
        print(f"    macro_f1        : {macro_f1.mean():.4f} ± {macro_f1.std(ddof=1):.4f}")
        print(f"    precision_fire  : {prec.mean():.4f} ± {prec.std(ddof=1):.4f}")
        print(f"    recall_fire     : {rec.mean():.4f} ± {rec.std(ddof=1):.4f}")


def section8():
    print("\n" + "=" * 70)
    print("SECTION 8: Sanity totals")
    print("=" * 70)
    total_runs = 0
    total_configs = 0
    for model in ("vit", "resnet", "hybrid"):
        configs = load_all(model)
        n_cfg = len(configs)
        n_runs = sum(len(v) for v in configs.values())
        print(f"  {model:8s}: {n_cfg:3d} configs, {n_runs:3d} runs")
        total_configs += n_cfg
        total_runs += n_runs
    print(f"  TOTAL: {total_configs} configs, {total_runs} runs (paper: 33 / 165)")
    # split sizes from any run
    r = load_runs("vit", "freeze_patch_blocks0-3")[0]
    ds = r["dataset"]
    print(f"  Splits: train={ds['train']['total']}  val={ds['val']['total']}  "
          f"test={ds['test']['total']}  (paper: 18847 / 2356 / 2356)")
    # param totals
    for model, cfg in [("vit", "freeze_none"), ("resnet", "freeze_none"),
                       ("hybrid", "freeze_none")]:
        runs = load_runs(model, cfg)
        if runs:
            tot = runs[0].get("num_total_params", "n/a")
            print(f"  {model} ({cfg}) total params = {tot:,}")


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section8()
