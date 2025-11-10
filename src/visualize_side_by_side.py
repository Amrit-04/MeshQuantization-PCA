#!/usr/bin/env python3
"""
visualize_side_by_side.py
----------------------------------------------------
Shows original (red) and reconstructed (green)
meshes side by side in Open3D.

Usage:
    python src/visualize_side_by_side.py --mesh branch --pipeline full
"""

import open3d as o3d
import argparse
import os

def load_mesh(path, color):
    if not os.path.exists(path):
        print(f"[!] Missing file: {path}")
        return None
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="mesh name, e.g. branch")
    parser.add_argument("--pipeline", choices=["core", "full"], required=True, help="which results folder to use")
    parser.add_argument("--method", default="unit_sphere", help="normalization method (unit_sphere / minmax)")
    parser.add_argument("--pca", default="pca", help="pca variant (pca / no_pca, ignored for core)")
    parser.add_argument("--bins", default="1024", help="quantization bins used (e.g. 1024)")
    args = parser.parse_args()

    mesh_name = args.mesh
    pipeline = args.pipeline

    orig_path = os.path.join("8samples", f"{mesh_name}.obj")
    results_dir = os.path.join(f"results_{pipeline}", mesh_name)
    if not os.path.isdir(results_dir):
        print(f"[!] Results directory missing: {results_dir}")
        return

    # Try to find the right reconstructed file automatically
    recon_file = None

    # Build ordered list of candidate substrings to match filenames
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

    # Search for matches in order of preference
    ply_files = [f for f in os.listdir(results_dir) if f.endswith(".ply")]
    for needle in candidates:
        for f in ply_files:
            if needle in f:
                recon_file = f
                break
        if recon_file:
            break

    # fallback: any ply file
    if recon_file is None and ply_files:
        recon_file = ply_files[0]

    if recon_file is None:
        print(f"[!] Could not find a matching reconstruction in {results_dir}")
        return

    recon_path = os.path.join(results_dir, recon_file)

    print(f"✅ Original:     {orig_path}")
    print(f"✅ Reconstruction: {recon_path}")

    orig = load_mesh(orig_path, [1, 0, 0])     # red
    recon = load_mesh(recon_path, [0, 1, 0])   # green

    if not orig or not recon:
        print("[!] Failed to load one or both meshes.")
        return

    # Compute offset for side-by-side placement
    bbox = orig.get_axis_aligned_bounding_box()
    width = bbox.get_extent()[0]
    recon.translate((width * 1.3, 0, 0))  # move right

    frame_orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=width * 0.2)
    frame_recon = o3d.geometry.TriangleMesh.create_coordinate_frame(size=width * 0.2)
    frame_recon.translate((width * 1.3, 0, 0))

    o3d.visualization.draw_geometries(
        [orig, recon, frame_orig, frame_recon],
        window_name=f"{mesh_name.upper()} — {pipeline.upper()} Pipeline (Side-by-Side)"
    )

if __name__ == "__main__":
    main()
