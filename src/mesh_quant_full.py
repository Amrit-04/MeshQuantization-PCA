#!/usr/bin/env python3
"""
mesh_quant_full.py — advanced version for MIXAR
Includes:
- Min–Max & Unit Sphere normalization
- PCA orientation + decorrelation
- Adaptive quantization (curvature-based)
- MSE / MAE / PSNR metrics & histograms

Input folder: 8samples/
Output folder: results_full/
"""

import os, numpy as np, trimesh, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- Normalization ----------
def minmax_normalize(v):
    vmin, vmax = v.min(0), v.max(0)
    denom = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
    return (v - vmin) / denom, {'method': 'minmax', 'vmin': vmin.tolist(), 'vmax': vmax.tolist()}

def unit_sphere_normalize(v):
    c = v.mean(0)
    centered = v - c
    r = np.linalg.norm(centered, axis=1).max()
    if r == 0: r = 1.0
    return centered / r, {'method': 'unit_sphere', 'centroid': c.tolist(), 'r': float(r)}

def denormalize(norm, meta):
    if meta['method'] == 'unit_sphere':
        return norm * meta['r'] + np.array(meta['centroid'])
    vmin = np.array(meta['vmin']); vmax = np.array(meta['vmax'])
    denom = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
    return norm * denom + vmin

# ---------- PCA ----------
def apply_pca(v):
    mean = v.mean(0)
    centered = v - mean
    pca = PCA(n_components=3)
    proj = pca.fit_transform(centered)
    return proj, pca, {'mean': mean.tolist(), 'components': pca.components_.tolist()}

def inverse_pca(proj, pca, meta):
    return pca.inverse_transform(proj) + np.array(meta['mean'])

# ---------- Quantization ----------
def quantize_uniform(norm, bins=1024):
    mn, mx = norm.min(), norm.max()
    scale = mx - mn if mx != mn else 1.0
    q = np.round((norm - mn) / scale * (bins - 1)).astype(int)
    return q, {'bins': bins, 'mn': float(mn), 'mx': float(mx)}

def dequantize_uniform(q, meta):
    b, mn, mx = meta['bins'], meta['mn'], meta['mx']
    scale = mx - mn if mx != mn else 1.0
    return q.astype(float) / (b - 1) * scale + mn

# ---------- Curvature Proxy ----------
def curvature_proxy(mesh):
    vnorms = mesh.vertex_normals
    adj = mesh.vertex_neighbors
    curv = np.zeros(len(vnorms))
    for i in range(len(vnorms)):
        neigh = adj[i]
        if len(neigh) == 0: continue
        avg = np.mean(vnorms[neigh], 0)
        curv[i] = np.linalg.norm(vnorms[i] - avg)
    return curv

# ---------- Adaptive Quantization ----------
def adaptive_quantize(v, curv, low=512, high=2048, th=80):
    T = np.percentile(curv, th)
    bins_per = np.where(curv >= T, high, low)
    mn, mx = v.min(), v.max()
    scale = mx - mn if mx != mn else 1.0
    mapped = (v - mn) / scale
    q = np.zeros_like(v, int)
    for i, b in enumerate(bins_per):
        q[i] = np.round(mapped[i] * (b - 1))
    return q, {'mn': float(mn), 'mx': float(mx), 'bins_per': bins_per.tolist()}

def adaptive_dequantize(q, meta):
    mn, mx = meta['mn'], meta['mx']
    scale = mx - mn if mx != mn else 1.0
    bins_per = np.array(meta['bins_per'])
    v = np.zeros_like(q, float)
    for i, b in enumerate(bins_per):
        v[i] = q[i] / (b - 1) * scale + mn
    return v

# ---------- Metrics ----------
def compute_error(orig, recon):
    diff = orig - recon
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    bbox = np.linalg.norm(orig.max(0) - orig.min(0))
    psnr = np.inf if mse == 0 else 10 * np.log10((bbox ** 2) / mse)
    return {'mse': mse, 'mae': mae, 'psnr': psnr}, diff

# ---------- Plot ----------
def plot_hist(diff, path):
    e = np.linalg.norm(diff, axis=1)
    plt.figure()
    plt.hist(e, bins=100)
    plt.title("Vertex L2 Error")
    plt.xlabel("Error Magnitude")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------- Main Process ----------
def process(path, out_dir, bins=[256,512,1024,2048]):
    mesh = trimesh.load(path, force='mesh')
    v = np.array(mesh.vertices, float)
    f = np.array(mesh.faces) if mesh.faces is not None else None
    name = os.path.splitext(os.path.basename(path))[0]
    mesh_dir = os.path.join(out_dir, name)
    ensure_dir(mesh_dir)

    # --- NEW: Print mesh statistics (Task 1) ---
    print(f"\n🔹 Mesh: {name}")
    print(f"Total vertices: {len(v)}")
    mins, maxs, means, stds = v.min(0), v.max(0), v.mean(0), v.std(0)
    print(f"  Min (x,y,z): ({mins[0]:.4f}, {mins[1]:.4f}, {mins[2]:.4f})")
    print(f"  Max (x,y,z): ({maxs[0]:.4f}, {maxs[1]:.4f}, {maxs[2]:.4f})")
    print(f"  Mean (x,y,z): ({means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f})")
    print(f"  Std  (x,y,z): ({stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f})")

    # --- Continue existing workflow ---
    curv = curvature_proxy(mesh)
    results = []

    for norm_func, nm in [(minmax_normalize, 'minmax'), (unit_sphere_normalize, 'unit_sphere')]:
        nv, meta = norm_func(v)

        # ----- Without PCA -----
        for b in bins:
            q, qm = quantize_uniform(nv, b)
            dq = dequantize_uniform(q, qm)
            recon = denormalize(dq, meta)
            m, d = compute_error(v, recon)
            results.append({'mesh': name, 'norm': nm, 'pca': 'no_pca', 'bins': b, **m})
            hist_path = os.path.join(mesh_dir, f"{name}_{nm}_no_pca_bins{b}_hist.png")
            plot_hist(d, hist_path)
            try:
                trimesh.Trimesh(vertices=recon, faces=f, process=False).export(
                    os.path.join(mesh_dir, f"{name}_{nm}_no_pca_bins{b}_recon.ply"))
            except:
                pass

        # Adaptive quantization (no PCA)
        q_a, qa = adaptive_quantize(nv, curv)
        dq_a = adaptive_dequantize(q_a, qa)
        recon = denormalize(dq_a, meta)
        m, d = compute_error(v, recon)
        results.append({'mesh': name, 'norm': nm, 'pca': 'no_pca', 'bins': 'adaptive', **m})
        plot_hist(d, os.path.join(mesh_dir, f"{name}_{nm}_no_pca_adaptive_hist.png"))
        try:
            trimesh.Trimesh(vertices=recon, faces=f, process=False).export(
                os.path.join(mesh_dir, f"{name}_{nm}_no_pca_adaptive_recon.ply"))
        except:
            pass

        # ----- With PCA -----
        nv2, pca, pca_meta = apply_pca(nv)
        for b in bins:
            q, qm = quantize_uniform(nv2, b)
            dq = dequantize_uniform(q, qm)
            recon = inverse_pca(dq, pca, pca_meta)
            recon = denormalize(recon, meta)
            m, d = compute_error(v, recon)
            results.append({'mesh': name, 'norm': nm, 'pca': 'pca', 'bins': b, **m})
            plot_hist(d, os.path.join(mesh_dir, f"{name}_{nm}_pca_bins{b}_hist.png"))
            try:
                trimesh.Trimesh(vertices=recon, faces=f, process=False).export(
                    os.path.join(mesh_dir, f"{name}_{nm}_pca_bins{b}_recon.ply"))
            except:
                pass

        # Adaptive with PCA
        q_a, qa = adaptive_quantize(nv2, curv)
        dq_a = adaptive_dequantize(q_a, qa)
        recon = inverse_pca(dq_a, pca, pca_meta)
        recon = denormalize(recon, meta)
        m, d = compute_error(v, recon)
        results.append({'mesh': name, 'norm': nm, 'pca': 'pca', 'bins': 'adaptive', **m})
        plot_hist(d, os.path.join(mesh_dir, f"{name}_{nm}_pca_adaptive_hist.png"))
        try:
            trimesh.Trimesh(vertices=recon, faces=f, process=False).export(
                os.path.join(mesh_dir, f"{name}_{nm}_pca_adaptive_recon.ply"))
        except:
            pass

    pd.DataFrame(results).to_csv(os.path.join(mesh_dir, f"{name}_metrics.csv"), index=False)
    return results

# ---------- Runner ----------
def main():
    in_dir = '8samples'
    out_dir = 'results_full'
    ensure_dir(out_dir)
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.lower().endswith('.obj')]
    if not files:
        print("⚠️ No .obj files found in", in_dir)
        return
    allr = []
    for f in tqdm(files, desc="Processing meshes"):
        allr += process(f, out_dir)
    pd.DataFrame(allr).to_csv(os.path.join(out_dir, 'all_metrics.csv'), index=False)
    print("\n✅ Done! Results saved to:", out_dir)

if __name__ == "__main__":
    main()
