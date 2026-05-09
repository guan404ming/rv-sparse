"""
Plot rv-sparse benchmark results.

Reads results/timings.csv (produced by bench/run_bench) and renders
three figures into results/:
  - density_sweep.png : runtime vs density at fixed 256x256
  - size_sweep.png    : runtime vs square dimension at fixed density 0.10
  - speedup_bar.png   : per-method speedup over M2_two_phase_naive at 256x256, density 0.10
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "timings.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

METHOD_ORDER = [
    "M1_dense_direct",
    "M2_two_phase_naive",
    "M3_two_phase_branchless",
    "M4_fused_branchy",
    "M5_fused_branchless",
]
METHOD_LABEL = {
    "M1_dense_direct":         "M1 dense direct (no CSR)",
    "M2_two_phase_naive":      "M2 two-phase, branchy",
    "M3_two_phase_branchless": "M3 two-phase, branchless (current)",
    "M4_fused_branchy":        "M4 fused, branchy",
    "M5_fused_branchless":     "M5 fused, branchless",
}
METHOD_COLOR = {
    "M1_dense_direct":         "#888888",
    "M2_two_phase_naive":      "#d62728",
    "M3_two_phase_branchless": "#1f77b4",
    "M4_fused_branchy":        "#ff7f0e",
    "M5_fused_branchless":     "#2ca02c",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["us_per_run"] = df["ns_per_run"] / 1000.0
    return df


def density_sweep(df: pd.DataFrame) -> None:
    sub = df[(df["rows"] == 256) & (df["cols"] == 256)].copy()
    sub = sub.groupby(["method", "density"], as_index=False)["us_per_run"].min()
    sub = sub.sort_values(["method", "density"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHOD_ORDER:
        s = sub[sub["method"] == m]
        ax.plot(s["density"] * 100, s["us_per_run"],
                marker="o", label=METHOD_LABEL[m], color=METHOD_COLOR[m], lw=1.6)
    ax.set_xlabel("density (% of non-zeros)")
    ax.set_ylabel("runtime per call (us)")
    ax.set_title("Runtime vs density, 256 x 256 (lower is better)")
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "density_sweep.png", dpi=140)
    plt.close(fig)


def size_sweep(df: pd.DataFrame) -> None:
    sub = df[(df["rows"] == df["cols"]) & np.isclose(df["density"], 0.10)].copy()
    sub = sub.groupby(["method", "rows"], as_index=False)["us_per_run"].min()
    sub = sub.sort_values(["method", "rows"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHOD_ORDER:
        s = sub[sub["method"] == m]
        ax.plot(s["rows"], s["us_per_run"],
                marker="s", label=METHOD_LABEL[m], color=METHOD_COLOR[m], lw=1.6)
    ax.set_xlabel("matrix dimension (N for N x N)")
    ax.set_ylabel("runtime per call (us)")
    ax.set_title("Runtime vs size, density = 10% (lower is better)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "size_sweep.png", dpi=140)
    plt.close(fig)


def speedup_bar(df: pd.DataFrame) -> None:
    sub = df[(df["rows"] == 256) & (df["cols"] == 256)
             & np.isclose(df["density"], 0.10)].copy()
    sub = sub.groupby("method", as_index=False)["ns_per_run"].min()
    base = float(sub.loc[sub["method"] == "M2_two_phase_naive", "ns_per_run"].iloc[0])

    speedups = []
    labels   = []
    colors   = []
    for m in METHOD_ORDER:
        ns = float(sub.loc[sub["method"] == m, "ns_per_run"].iloc[0])
        speedups.append(base / ns)
        labels.append(METHOD_LABEL[m])
        colors.append(METHOD_COLOR[m])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, speedups, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("speedup vs M2 two-phase naive")
    ax.set_title("Speedup at 256 x 256, density = 10% (higher is better)")
    for b, v in zip(bars, speedups):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}x",
                ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "speedup_bar.png", dpi=140)
    plt.close(fig)


def summary(df: pd.DataFrame) -> None:
    sub = df[(df["rows"] == 256) & (df["cols"] == 256)].copy()
    pivot = sub.pivot_table(index="density", columns="method",
                            values="ns_per_run", aggfunc="min")
    pivot = pivot[METHOD_ORDER]
    print("\n--- runtime (ns/call) at 256x256 ---")
    print(pivot.round(1).to_string())

    sub2 = df[(df["rows"] == df["cols"]) & np.isclose(df["density"], 0.10)].copy()
    pivot2 = sub2.pivot_table(index="rows", columns="method",
                              values="ns_per_run", aggfunc="min")
    pivot2 = pivot2[METHOD_ORDER]
    print("\n--- runtime (ns/call) at density=0.10, NxN ---")
    print(pivot2.round(1).to_string())


def main() -> None:
    df = load()
    density_sweep(df)
    size_sweep(df)
    speedup_bar(df)
    summary(df)
    print(f"\nfigures written under {OUT}")


if __name__ == "__main__":
    main()
