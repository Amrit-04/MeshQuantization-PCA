#!/usr/bin/env python3
"""
visualize_bins_grid.py
----------------------------------------------------
Displays a 2x2 grid of reconstructions for one mesh
at different quantization bin levels.

Usage:
    python src/visualize_bins_grid.py --mesh girl --method unit_sphere --pca pca
"""

import open3d as o3d
import argparse
import os

def load_mesh(path, color):
    if not os.path.exists(path):
        print(f"[!] Missing: {path}")
        return None
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Mesh name (e.g., girl, branch, talwar)")
    parser.add_argument("--pipeline", default="full", help="Pipeline folder (default=full)")
    parser.add_argument("--method", default="unit_sphere", help="Normalization method (minmax/unit_sphere)")
    parser.add_argument("--pca", default="pca", help="PCA variant (pca/no_pca)")
    args = parser.parse_args()

    bins = [256, 512, 1024, 2048]
    base_dir = os.path.join(f"results_{args.pipeline}", args.mesh)
    orig_path = os.path.join("8samples", f"{args.mesh}.obj")

    if not os.path.exists(orig_path):
        print(f"[!] Original mesh missing: {orig_path}")
        return

    if not os.path.isdir(base_dir):
        print(f"[!] Results folder missing: {base_dir}")
        return

    orig = load_mesh(orig_path, [1, 0, 0])  # red (original)
    if orig is None:
        return

    recon_meshes = []
    ply_files = [f for f in os.listdir(base_dir) if f.endswith(".ply")]

    for b in bins:
        recon_file = None
        candidates = []
        if args.pipeline == "full":
            candidates.append(f"{args.mesh}_{args.method}_{args.pca}_bins{b}_recon")
            candidates.append(f"{args.method}_{args.pca}_bins{b}_recon")
        candidates.append(f"{args.mesh}_{args.method}_bins{b}_recon")
        candidates.append(f"{args.method}_bins{b}_recon")
        candidates.append(f"bins{b}_recon")

        for needle in candidates:
            for f in ply_files:
                if needle in f:
                    recon_file = f
                    break
            if recon_file:
                break

        if recon_file is None:
            print(f"[!] Missing reconstruction for {b} bins")
            continue

        fpath = os.path.join(base_dir, recon_file)
        mesh = load_mesh(fpath, [0, 1, 0])
        if mesh:
            print(f"✅ Loaded reconstruction for {b} bins: {recon_file}")
            recon_meshes.append((b, mesh))

    if not recon_meshes:
        print("[!] No meshes found to visualize.")
        return

    # Compute spacing based on bounding box
    bbox = orig.get_axis_aligned_bounding_box()
    dx = bbox.get_extent()[0] * 1.8
    dz = bbox.get_extent()[2] * 1.8

    # Arrange meshes in 2x2 grid (256,512,1024,2048)
    placed = []
    grid_positions = {
        256: (-dx, 0, dz),
        512: (dx, 0, dz),
        1024: (-dx, 0, -dz),
        2048: (dx, 0, -dz)
    }

    for b, mesh in recon_meshes:
        pos = grid_positions.get(b, (0, 0, 0))
        mesh.translate(pos)
        placed.append(mesh)

    # Add original mesh at center
    orig.paint_uniform_color([1, 0, 0])
    placed.append(orig)

    # Create coordinate frame for scale reference
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=bbox.get_extent()[0] * 0.3)
    placed.append(frame)

    # Show window
    o3d.visualization.draw_geometries(
        placed,
        window_name=f"{args.mesh.upper()} — Quantization Comparison (256→2048 bins)",
        width=1600,
        height=900
    )

if __name__ == "__main__":
    main()

