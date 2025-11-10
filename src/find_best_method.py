#!/usr/bin/env python3
"""
find_best_method.py
----------------------------------------------------
Analyzes all results in results_full/all_metrics.csv
and identifies the best combination of:
  - Normalization (minmax / unit_sphere)
  - PCA (pca / no_pca)
  - Quantization bins

Outputs:
  results_comparison/best_method_summary.csv
  + prints best methods to terminal
"""

import pandas as pd
import os

def main():
    full_path = "results_full/all_metrics.csv"
    out_dir = "results_comparison"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(full_path):
        print(f"❌ File not found: {full_path}")
        return

    df = pd.read_csv(full_path)
    required = {"norm", "pca", "bins", "mse", "psnr"}
    if not required.issubset(df.columns):
        print("❌ Missing required columns in CSV.")
        return

    # Remove adaptive rows if needed
    df_clean = df[df["bins"].astype(str).str.isnumeric()].copy()
    df_clean["bins"] = df_clean["bins"].astype(int)

    # Group and average errors
    grouped = (
        df_clean.groupby(["norm", "pca", "bins"])[["mse", "mae", "psnr"]]
        .mean()
        .reset_index()
        .sort_values("mse")
    )

    # Identify best by MSE and PSNR
    best_mse_row = grouped.loc[grouped["mse"].idxmin()]
    best_psnr_row = grouped.loc[grouped["psnr"].idxmax()]

    print("\n=== 🔍 Best Combination by MSE ===")
    print(best_mse_row.to_string(index=False))

    print("\n=== 🔍 Best Combination by PSNR ===")
    print(best_psnr_row.to_string(index=False))

    # Save summary
    grouped.to_csv(os.path.join(out_dir, "best_method_summary.csv"), index=False)
    print("\n✅ Saved detailed averages to results_comparison/best_method_summary.csv")

if __name__ == "__main__":
    main()
