#!/usr/bin/env python3
"""
compare_with_original.py — Compare reconstructed meshes vs original .obj files

Usage:
    python src/compare_with_original.py --pipeline core
    python src/compare_with_original.py --pipeline full
"""

import os
import argparse
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Helper ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def compute_error(orig_v, recon_v):
    if orig_v.shape != recon_v.shape:
        print("⚠️ Vertex count mismatch, truncating to shortest length")
        n = min(len(orig_v), len(recon_v))
        orig_v, recon_v = orig_v[:n], recon_v[:n]
    diff = orig_v - recon_v
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    bbox = np.linalg.norm(orig_v.max(0) - orig_v.min(0))
    psnr = np.inf if mse == 0 else 10 * np.log10((bbox ** 2) / mse)
    return {'mse': mse, 'mae': mae, 'psnr': psnr}, diff

def plot_hist(diff, out_path):
    e = np.linalg.norm(diff, axis=1)
    plt.figure(figsize=(6,4))
    plt.hist(e, bins=100, color="purple", alpha=0.7)
    plt.title("Vertex Error Distribution")
    plt.xlabel("L2 Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["core", "full"], required=True,
                        help="Which results folder to compare: core or full")
    args = parser.parse_args()

    in_dir = "8samples"
    results_dir = f"results_{args.pipeline}"
    out_dir = f"results_comparison/original_vs_{args.pipeline}"
    ensure_dir(out_dir)

    if not os.path.exists(results_dir):
        print(f"❌ {results_dir} not found. Run that pipeline first.")
        return

    meshes = [f for f in os.listdir(in_dir) if f.lower().endswith(".obj")]
    all_results = []

    print(f"\n🔍 Comparing {args.pipeline.upper()} reconstructions against originals...\n")

    for m in tqdm(meshes):
        orig_path = os.path.join(in_dir, m)
        name = os.path.splitext(m)[0]
        recon_dir = os.path.join(results_dir, name)
        if not os.path.exists(recon_dir):
            print(f"⚠️ No reconstruction folder for {name}, skipping")
            continue

        # find all .ply reconstructions in this folder
        recon_files = [f for f in os.listdir(recon_dir) if f.endswith(".ply")]
        if not recon_files:
            print(f"⚠️ No .ply files for {name}, skipping")
            continue

        orig_mesh = trimesh.load(orig_path, force='mesh')
        orig_v = np.array(orig_mesh.vertices, float)

        for rf in recon_files:
            recon_path = os.path.join(recon_dir, rf)
            recon_mesh = trimesh.load(recon_path, force='mesh')
            recon_v = np.array(recon_mesh.vertices, float)

            metrics, diff = compute_error(orig_v, recon_v)
            metrics['mesh'] = name
            metrics['file'] = rf
            all_results.append(metrics)

            # Save error histogram
            plot_hist(diff, os.path.join(out_dir, f"{name}_{rf.replace('.ply','')}_errorhist.png"))

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, "error_vs_original.csv")
    df.to_csv(csv_path, index=False)

    print("\n✅ Done! Results saved to:", csv_path)
    print(df.groupby("mesh")[["mse","mae","psnr"]].mean())

if __name__ == "__main__":
    main()
