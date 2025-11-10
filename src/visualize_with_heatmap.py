#!/usr/bin/env python3
"""
visualize_with_heatmap.py
----------------------------------------------------
Shows:
    - Original mesh (red, semi-transparent)
    - Reconstructed mesh (green)
    - Error heatmap (.ply from results_comparison/error_heatmaps/)

Usage:
    python src/visualize_with_heatmap.py --mesh branch --pipeline full
"""

import open3d as o3d
import argparse
import os

def load_mesh(path, color=None, alpha=1.0):
    if not os.path.exists(path):
        print(f"[!] Missing: {path}")
        return None
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Mesh name (e.g. branch)")
    parser.add_argument("--pipeline", choices=["core", "full"], required=True,
                        help="Which pipeline to visualize (core or full)")
    parser.add_argument("--method", default="unit_sphere",
                        help="Normalization method (unit_sphere/minmax)")
    parser.add_argument("--bins", default="1024", help="Quantization bins (e.g. 1024)")
    parser.add_argument("--pca", default="pca", help="PCA variant (pca / no_pca, ignored for core)")
    args = parser.parse_args()

    mesh_name = args.mesh
    pipeline = args.pipeline

    # Define paths
    orig_path = os.path.join("8samples", f"{mesh_name}.obj")
    recon_dir = os.path.join(f"results_{pipeline}", mesh_name)
    heatmap_dir = os.path.join("results_comparison", "error_heatmaps")

    if not os.path.exists(orig_path):
        print(f"[!] Original mesh not found: {orig_path}")
        return

    if not os.path.exists(recon_dir):
        print(f"[!] No reconstruction folder for {mesh_name}")
        return

    # Auto-detect reconstructed .ply
    recon_file = None
    ply_files = [f for f in os.listdir(recon_dir) if f.endswith(".ply")]

    candidates = []
    if pipeline == "full":
        if args.method and args.pca and args.bins:
            candidates.append(f"{args.method}_{args.pca}_bins{args.bins}")
            candidates.append(f"{args.method}_{args.pca}_{args.bins}")
        if args.method and args.pca:
            candidates.append(f"{args.method}_{args.pca}")
    if args.method and args.bins:
        candidates.append(f"{args.method}_bins{args.bins}")
    if args.method:
        candidates.append(args.method)
    if args.bins:
        candidates.append(str(args.bins))

    for needle in candidates:
        for f in ply_files:
            if needle in f:
                recon_file = f
                break
        if recon_file:
            break

    if recon_file is None and ply_files:
        recon_file = ply_files[0]

    if recon_file is None:
        print(f"[!] Could not find reconstructed mesh in {recon_dir}")
        return

    recon_path = os.path.join(recon_dir, recon_file)

    # Auto-detect heatmap file
    heatmap_file = None
    expected_heatmap = None
    if recon_file:
        recon_stub = recon_file.replace(".ply", "")
        expected_heatmap = f"{mesh_name}_{pipeline}_{recon_stub}_heatmap.ply"
        candidate_path = os.path.join(heatmap_dir, expected_heatmap)
        if os.path.exists(candidate_path):
            heatmap_file = expected_heatmap

    if heatmap_file is None and os.path.isdir(heatmap_dir):
        for f in os.listdir(heatmap_dir):
            if f.endswith(".ply") and mesh_name in f and pipeline in f:
                heatmap_file = f
                break

    heatmap_path = os.path.join(heatmap_dir, heatmap_file) if heatmap_file else None

    print(f"\n✅ Original:      {orig_path}")
    print(f"✅ Reconstruction: {recon_path}")
    print(f"🎨 Heatmap:       {heatmap_path if heatmap_path else 'Not found'}\n")

    # Load meshes
    orig = load_mesh(orig_path, [1, 0, 0])     # red
    recon = load_mesh(recon_path, [0, 1, 0])   # green
    heatmap = load_mesh(heatmap_path) if heatmap_path else None

    if not orig or not recon:
        print("[!] Failed to load one or both main meshes.")
        return

    # Make original slightly transparent and scale reference frame
    bbox = orig.get_axis_aligned_bounding_box()
    width = bbox.get_extent()[0]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=width * 0.3)

    meshes = [orig, recon, frame]
    if heatmap:
        meshes.append(heatmap)

    print("🟢 Open3D viewer: Use mouse to rotate, scroll to zoom, R to reset")
    o3d.visualization.draw_geometries(
        meshes,
        window_name=f"{mesh_name.upper()} — {pipeline.upper()} Pipeline + Heatmap"
    )

if __name__ == "__main__":
    main()
