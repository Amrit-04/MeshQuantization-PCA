#!/usr/bin/env python3
"""
error_summary_and_heatmaps.py
---------------------------------------------------
Generates:
  1. Combined error summary CSV comparing Core & Full pipelines.
  2. Per-vertex error heatmaps (.ply) showing error vs original.

Usage:
  python src/error_summary_and_heatmaps.py
"""

import os
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Helper ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def compute_error(orig_v, recon_v):
    if orig_v.shape != recon_v.shape:
        n = min(len(orig_v), len(recon_v))
        orig_v, recon_v = orig_v[:n], recon_v[:n]
    diff = orig_v - recon_v
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    bbox = np.linalg.norm(orig_v.max(0) - orig_v.min(0))
    psnr = np.inf if mse == 0 else 10 * np.log10((bbox ** 2) / mse)
    return {'mse': mse, 'mae': mae, 'psnr': psnr}, diff

def colorize_error(mesh, diff):
    # Compute normalized per-vertex error
    errs = np.linalg.norm(diff, axis=1)
    if errs.max() > 0:
        errs_norm = (errs - errs.min()) / (errs.max() - errs.min())
    else:
        errs_norm = errs
    # Convert to RGB heatmap: blue → red
    cmap = plt.get_cmap("jet")
    colors = (cmap(errs_norm)[:, :3] * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    return mesh

# ---------- Main ----------
def main():
    in_dir = "8samples"
    results_dirs = {"core": "results_core", "full": "results_full"}
    out_dir = "results_comparison"
    ensure_dir(out_dir)
    heatmap_dir = os.path.join(out_dir, "error_heatmaps")
    ensure_dir(heatmap_dir)

    all_records = []

    print("\n🔍 Generating global error summary and heatmaps...\n")

    for label, res_dir in results_dirs.items():
        if not os.path.exists(res_dir):
            print(f"⚠️ Skipping {label} — folder not found: {res_dir}")
            continue

        meshes = [f for f in os.listdir(in_dir) if f.lower().endswith(".obj")]

        for m in tqdm(meshes, desc=f"Comparing {label.upper()} pipeline"):
            orig_path = os.path.join(in_dir, m)
            name = os.path.splitext(m)[0]
            recon_dir = os.path.join(res_dir, name)
            if not os.path.exists(recon_dir):
                continue

            # Get all reconstructions for that mesh
            recon_files = [f for f in os.listdir(recon_dir) if f.endswith(".ply")]
            if not recon_files:
                continue

            orig_mesh = trimesh.load(orig_path, force='mesh')
            orig_v = np.array(orig_mesh.vertices, float)

            for rf in recon_files:
                recon_path = os.path.join(recon_dir, rf)
                recon_mesh = trimesh.load(recon_path, force='mesh')
                recon_v = np.array(recon_mesh.vertices, float)

                metrics, diff = compute_error(orig_v, recon_v)
                metrics.update({"mesh": name, "pipeline": label, "file": rf})
                all_records.append(metrics)

                # Save heatmap
                colored_mesh = colorize_error(recon_mesh.copy(), diff)
                heatmap_out = os.path.join(
                    heatmap_dir, f"{name}_{label}_{rf.replace('.ply','')}_heatmap.ply"
                )
                colored_mesh.export(heatmap_out)

    df = pd.DataFrame(all_records)
    if len(df) == 0:
        print("❌ No results found to summarize.")
        return

    summary = (
        df.groupby(["mesh", "pipeline"])[["mse", "mae", "psnr"]]
        .mean()
        .reset_index()
        .sort_values(["mesh", "mse"])
    )

    csv_path = os.path.join(out_dir, "error_summary.csv")
    summary.to_csv(csv_path, index=False)

    print("\n✅ Global summary saved to:", csv_path)
    print(summary)

    print("\n✅ Heatmaps saved under:", heatmap_dir)
    print("Open them in MeshLab or Open3D to visualize per-vertex error.")

if __name__ == "__main__":
    main()
