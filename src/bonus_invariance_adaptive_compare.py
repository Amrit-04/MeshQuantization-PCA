#!/usr/bin/env python3
"""
bonus_invariance_adaptive_compare.py
----------------------------------------------------
Rotation + Translation Invariance + Adaptive Quantization
Comparison between Unit Sphere and PCA normalization.
"""

import trimesh, numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from tqdm import tqdm
from sklearn.decomposition import PCA

# -------------------- UTILITIES --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def random_transform(vertices):
    """Apply random rotation + translation."""
    rand = np.random.randn(3, 3)
    u, _, v = np.linalg.svd(rand)
    R = u @ v
    t = np.random.uniform(-0.5, 0.5, (1, 3))
    return vertices @ R.T + t, R, t

def unit_sphere_normalize(v):
    c = v.mean(0)
    centered = v - c
    r = np.linalg.norm(centered, axis=1).max()
    if r == 0: r = 1.0
    return centered / r, {'centroid': c, 'r': r}

def pca_normalize(v):
    mean = v.mean(0)
    pca = PCA(n_components=3)
    proj = pca.fit_transform(v - mean)
    scale = np.abs(proj).max()
    return proj / scale, {'mean': mean, 'components_': pca.components_, 'scale': scale}

def denormalize_pca(v, meta):
    pca = np.array(meta['components_'])
    mean = np.array(meta['mean'])
    return (v * meta['scale']) @ pca + mean

def adaptive_quantize(v, curv, low=512, high=2048, th=80):
    T = np.percentile(curv, th)
    bins = np.where(curv >= T, high, low)
    mn, mx = v.min(), v.max()
    scale = mx - mn if mx != mn else 1.0
    mapped = (v - mn) / scale
    q = np.zeros_like(v, int)
    for i, b in enumerate(bins): q[i] = np.round(mapped[i] * (b - 1))
    return q, {'mn': mn, 'mx': mx, 'bins': bins}

def dequantize(q, meta):
    mn, mx, bins = meta['mn'], meta['mx'], meta['bins']
    scale = mx - mn if mx != mn else 1.0
    v = np.zeros_like(q, float)
    for i, b in enumerate(bins): v[i] = q[i] / (b - 1) * scale + mn
    return v

def mse(a,b): return np.mean((a-b)**2)

def curvature_proxy(mesh):
    n = mesh.vertex_normals
    adj = mesh.vertex_neighbors
    curv = np.zeros(len(n))
    for i in range(len(n)):
        if len(adj[i]) == 0: continue
        curv[i] = np.linalg.norm(n[i] - np.mean(n[adj[i]], 0))
    return curv

# -------------------- MAIN --------------------
def main():
    in_dir, out_dir = "8samples", "results_bonus"
    ensure_dir(out_dir)
    meshes = [f for f in os.listdir(in_dir) if f.endswith(".obj")]
    results = []

    for name in tqdm(meshes, desc="Meshes"):
        mesh = trimesh.load(os.path.join(in_dir, name))
        v = np.array(mesh.vertices, float)
        f = np.array(mesh.faces)
        curv = curvature_proxy(mesh)

        mesh_outdir = os.path.join(out_dir, os.path.splitext(name)[0])
        ensure_dir(mesh_outdir)

        for i in range(3):  # 3 random rotations/translations
            vt, R, t = random_transform(v)

            # Save transformed mesh
            trans_path = os.path.join(mesh_outdir, f"{name.replace('.obj','')}_rot{i}.ply")
            trimesh.Trimesh(vertices=vt, faces=f, process=False).export(trans_path)

            for norm_type in ["unit_sphere", "pca"]:
                if norm_type == "unit_sphere":
                    norm_v, meta = unit_sphere_normalize(vt)
                else:
                    norm_v, meta = pca_normalize(vt)

                # Adaptive vs uniform quantization
                q_a, qa = adaptive_quantize(norm_v, curv)
                dq_a = dequantize(q_a, qa)
                mse_adapt = mse(norm_v, dq_a)

                q_u, _ = adaptive_quantize(norm_v, np.zeros_like(curv))
                dq_u = dequantize(q_u, qa)
                mse_uni = mse(norm_v, dq_u)

                # Save reconstructions
                trimesh.Trimesh(vertices=dq_a, faces=f, process=False).export(
                    os.path.join(mesh_outdir, f"{name.replace('.obj','')}_rot{i}_{norm_type}_adaptive.ply"))
                trimesh.Trimesh(vertices=dq_u, faces=f, process=False).export(
                    os.path.join(mesh_outdir, f"{name.replace('.obj','')}_rot{i}_{norm_type}_uniform.ply"))

                results.append({
                    "mesh": name,
                    "trial": i,
                    "normalization": norm_type,
                    "mse_adaptive": mse_adapt,
                    "mse_uniform": mse_uni
                })

    # Save metrics
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "rotation_invariance_compare.csv"), index=False)

    # Plot comparison per normalization
    plt.figure(figsize=(8,5))
    for norm_type in ["unit_sphere", "pca"]:
        mean_df = df[df["normalization"] == norm_type].groupby("mesh")[["mse_adaptive","mse_uniform"]].mean()
        mean_df["mse_adaptive"].plot(kind="bar", alpha=0.7, label=f"{norm_type} - adaptive")
        mean_df["mse_uniform"].plot(kind="bar", alpha=0.7, label=f"{norm_type} - uniform")
    plt.title("Adaptive vs Uniform Quantization Error\n(Unit Sphere vs PCA Normalization)")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adaptive_vs_uniform_compare.png"), dpi=150)
    plt.show()

    print("✅ All transformed and reconstructed meshes saved under results_bonus/")
    print("✅ Metrics saved as rotation_invariance_compare.csv")

if __name__ == "__main__":
    main()
