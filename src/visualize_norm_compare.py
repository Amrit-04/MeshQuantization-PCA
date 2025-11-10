#!/usr/bin/env python3
"""
visualize_norm_compare.py
----------------------------------------------------
Visualize Unit Sphere vs PCA normalization results
for the same rotated mesh (adaptive quantization).

Usage:
    python src/visualize_norm_compare.py --mesh girl --rotation 0
"""

import open3d as o3d
import os, argparse

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
    parser.add_argument("--mesh", required=True, help="Mesh name (e.g., girl, branch)")
    parser.add_argument("--rotation", type=int, default=0, help="Rotation index (0,1,2)")
    args = parser.parse_args()

    mesh_name = args.mesh
    rot_idx = args.rotation
    base_dir = os.path.join("results_bonus", mesh_name)

    # File paths
    orig_path = os.path.join(base_dir, f"{mesh_name}_rot{rot_idx}.ply")
    adaptive_path = os.path.join(base_dir, f"{mesh_name}_rot{rot_idx}_adaptive.ply")
    uniform_path = os.path.join(base_dir, f"{mesh_name}_rot{rot_idx}_uniform.ply")

    print(f"✅ Original: {orig_path}")
    print(f"✅ Adaptive: {adaptive_path}")
    print(f"✅ Uniform: {uniform_path}")

    # Load meshes
    orig = load_mesh(orig_path, [1, 0, 0])          # red
    adaptive = load_mesh(adaptive_path, [0, 1, 0])   # green
    uniform = load_mesh(uniform_path, [0, 0.5, 1])   # blue

    if not (orig and adaptive and uniform):
        print("[!] Missing one or more meshes. Check your paths.")
        return

    # Compute spacing
    bbox = orig.get_axis_aligned_bounding_box()
    dx = bbox.get_extent()[0] * 2.2

    # Position them side by side
    adaptive.translate((-dx, 0, 0))   # left
    uniform.translate((dx, 0, 0))     # right

    # Labels using coordinate frames
    frame_orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=dx*0.3)
    frame_adaptive = o3d.geometry.TriangleMesh.create_coordinate_frame(size=dx*0.3).translate((-dx, 0, 0))
    frame_uniform = o3d.geometry.TriangleMesh.create_coordinate_frame(size=dx*0.3).translate((dx, 0, 0))

    print("\n🟢 Controls:")
    print("   Rotate view: drag with left mouse")
    print("   Zoom: scroll")
    print("   R: reset view")
    print("   Q/Esc: quit viewer\n")

    o3d.visualization.draw_geometries(
        [orig, adaptive, uniform, frame_orig, frame_adaptive, frame_uniform],
        window_name=f"{mesh_name.upper()} — Rotation {rot_idx}: Original (Red) | Adaptive (Green) | Uniform (Blue)",
        width=1600,
        height=900
    )

if __name__ == "__main__":
    main()
