#!/usr/bin/env python3
"""
plot_error_per_axis.py
----------------------------------------------------
Plots reconstruction error (MSE) per coordinate axis (x, y, z)
for each normalization and quantization configuration.

Fixes:
- Handles meshes with mismatched vertex counts (align by nearest neighbor)
- Skips invalid files gracefully

Outputs:
- results_comparison/error_per_axis.csv
- results_comparison/error_per_axis_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import cKDTree

# --- Directories ---
in_dir = "8samples"
recon_dir = "results_full"
out_dir = "results_comparison"
os.makedirs(out_dir, exist_ok=True)

# --- Helpers ---
def mse_per_axis(orig, recon):
    """Compute per-axis MSE"""
    diff = orig - recon
    return np.mean(diff ** 2, axis=0)  # returns [MSE_x, MSE_y, MSE_z]

def align_by_nearest(orig, recon):
    """Align meshes by nearest neighbor if vertex counts differ"""
    tree = cKDTree(recon)
    _, idx = tree.query(orig, k=1)
    return recon[idx]

# --- Main process ---
mesh_files = [f for f in os.listdir(in_dir) if f.endswith(".obj")]
all_rows = []

for mesh_file in mesh_files:
    name = os.path.splitext(mesh_file)[0]
    orig_mesh = trimesh.load(os.path.join(in_dir, mesh_file), force='mesh')
    v_orig = np.array(orig_mesh.vertices)

    subdir = os.path.join(recon_dir, name)
    if not os.path.isdir(subdir):
        continue

    print(f"\n🔹 Processing mesh: {name}")
    for f in os.listdir(subdir):
        if not f.endswith("_recon.ply"):
            continue

        fpath = os.path.join(subdir, f)
        parts = f.replace(".ply", "").split("_")
        if len(parts) < 5:
            continue
        norm = parts[1]
        pca = parts[2]
        bins = parts[-1].replace("bins", "")

        try:
            recon_mesh = trimesh.load(fpath, force='mesh')
            v_recon = np.array(recon_mesh.vertices)

            # align mismatched meshes
            if v_orig.shape[0] != v_recon.shape[0]:
                print(f"⚠️ Aligning by nearest neighbor: {f} ({v_orig.shape[0]} vs {v_recon.shape[0]})")
                v_recon = align_by_nearest(v_orig, v_recon)

            mse_x, mse_y, mse_z = mse_per_axis(v_orig, v_recon)
            all_rows.append({
                "mesh": name,
                "norm": norm,
                "pca": pca,
                "bins": bins,
                "MSE_X": mse_x,
                "MSE_Y": mse_y,
                "MSE_Z": mse_z
            })
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")
            continue

# --- Save results ---
df = pd.DataFrame(all_rows)
if df.empty:
    print("⚠️ No valid reconstructions found.")
    exit()

csv_path = os.path.join(out_dir, "error_per_axis.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Saved CSV: {csv_path}")

# --- Plot average per-axis MSE ---
mean_df = df.groupby("bins")[["MSE_X", "MSE_Y", "MSE_Z"]].mean()

plt.figure(figsize=(7, 4))
for axis, color in zip(["MSE_X", "MSE_Y", "MSE_Z"], ["r", "g", "b"]):
    plt.plot(mean_df.index.astype(str), mean_df[axis], marker='o', label=axis, color=color)

plt.xlabel("Quantization Bins")
plt.ylabel("Mean Squared Error")
plt.title("Reconstruction Error per Axis (Averaged over all meshes)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plot_path = os.path.join(out_dir, "error_per_axis_plot.png")
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"✅ Saved plot: {plot_path}")
