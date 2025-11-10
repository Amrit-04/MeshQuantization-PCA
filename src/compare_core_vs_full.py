#!/usr/bin/env python3
"""
compare_core_vs_full.py — Error Analysis for MIXAR
--------------------------------------------------
Reads results from:
    results_core/all_metrics.csv
    results_full/all_metrics.csv

Computes:
    - Average MSE, MAE, PSNR per method
    - Best configuration per mesh
    - Comparative rate–distortion plots
    - Bar plots comparing Core vs Full performance

Usage:
    python src/compare_core_vs_full.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------ Load Data ------------------
def load_results():
    core_path = "results_core/all_metrics.csv"
    full_path = "results_full/all_metrics.csv"

    if not os.path.exists(core_path):
        raise FileNotFoundError(f"❌ Missing {core_path}. Run mesh_quant_core.py first.")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Missing {full_path}. Run mesh_quant_full.py first.")

    core = pd.read_csv(core_path)
    full = pd.read_csv(full_path)

    core["pipeline"] = "core"
    full["pipeline"] = "full"

    return core, full


# ------------------ Basic Summaries ------------------
def summarize_basic(df, label):
    print(f"\n🔹 {label.upper()} PIPELINE SUMMARY")
    print("=" * 50)

    if "psnr" in df.columns:
        cols = ["mse", "mae", "psnr"]
    else:
        cols = ["mse", "mae"]

    avg = df[cols].mean()
    print(f"Average Errors:\n{avg}\n")

    if "norm" in df.columns:
        print("Average per Normalization Method:")
        print(df.groupby("norm")[cols].mean().sort_values("mse"))
    elif "method" in df.columns:
        print("Average per Method:")
        print(df.groupby("method")[cols].mean().sort_values("mse"))

    print("=" * 50)


# ------------------ Comparison Summary ------------------
def compare_pipelines(core, full):
    print("\n🧮 GLOBAL COMPARISON SUMMARY")
    print("=" * 50)

    core_mean = core["mse"].mean()
    full_mean = full["mse"].mean()
    diff = (core_mean - full_mean) / core_mean * 100

    print(f"Average MSE (Core):  {core_mean:.6f}")
    print(f"Average MSE (Full):  {full_mean:.6f}")
    print(f"🔻 Improvement:       {diff:.2f}% lower error in Full version")

    if "psnr" in full.columns:
        print(f"Average PSNR (Full): {full['psnr'].mean():.2f} dB")

    print("=" * 50)


# ------------------ Visualization ------------------
def plot_comparison(core, full):
    out_dir = "results_comparison"
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Histogram of MSEs ---
    plt.figure(figsize=(8,5))
    plt.hist(core["mse"], bins=30, alpha=0.6, label="Core", color="orange")
    plt.hist(full["mse"], bins=30, alpha=0.6, label="Full", color="green")
    plt.title("MSE Distribution — Core vs Full Pipelines")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mse_histogram.png"), dpi=150)
    plt.close()

    # --- 2. Average MSE per Normalization ---
    if "norm" in full.columns:
        avg_full = full.groupby("norm")["mse"].mean()
        plt.figure(figsize=(7,4))
        avg_full.plot(kind="bar", color="seagreen")
        plt.ylabel("Average MSE")
        plt.title("Average MSE per Normalization (Full)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mse_by_norm_full.png"), dpi=150)
        plt.close()

    # --- 3. Rate–Distortion Curve (bins vs MSE) ---
    if "bins" in full.columns:
        valid = full[full["bins"].apply(lambda x: str(x).isdigit())]
        valid["bins"] = valid["bins"].astype(int)
        plt.figure(figsize=(8,5))
        for nm, g in valid.groupby("norm"):
            plt.plot(g["bins"], g["mse"], "o-", label=nm)
        plt.xlabel("Quantization bins")
        plt.ylabel("Mean Squared Error")
        plt.title("Rate–Distortion Curve (Full Pipeline)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rate_distortion_full.png"), dpi=150)
        plt.close()

    print(f"✅ Plots saved to: {out_dir}/")


# ------------------ Entry Point ------------------
def main():
    core, full = load_results()

    summarize_basic(core, "core")
    summarize_basic(full, "full")
    compare_pipelines(core, full)
    plot_comparison(core, full)

    print("\n✅ Comparison complete! Check 'results_comparison/' for plots and visual summaries.")

if __name__ == "__main__":
    main()
