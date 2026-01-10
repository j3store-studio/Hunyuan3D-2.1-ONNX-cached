"""
Mesh utilities for Hunyuan3D - Subprocess bpy wrapper

Calls bpy operations via subprocess using a separate Python 3.10 environment
since bpy doesn't support Python 3.12.

The actual bpy operations are in /app/bpy_mesh_ops.py which runs in /opt/bpy-env/
"""

import os
import subprocess
import tempfile
import trimesh
from typing import Optional

# Path to the bpy environment and wrapper script
BPY_PYTHON = "/opt/bpy-env/bin/python"
BPY_SCRIPT = "/app/bpy_mesh_ops.py"


def convert_obj_to_glb(
    input_path: str,
    output_path: str,
    apply_vertex_merge: bool = True,
    shading_mode: str = "auto",  # "smooth", "flat", "auto"
    auto_smooth_angle: float = 30.0
) -> str:
    """
    Convert OBJ file to GLB format using Blender via subprocess.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path for output GLB file
        apply_vertex_merge: Whether to merge duplicate vertices
        shading_mode: Shading mode - "smooth", "flat", or "auto"
        auto_smooth_angle: Angle for auto-smooth (degrees)
    
    Returns:
        Path to output GLB file
    """
    # Build command
    cmd = [
        BPY_PYTHON,
        BPY_SCRIPT,
        "convert_obj_to_glb",
        input_path,
        output_path,
        f"--shading={shading_mode}",
        f"--angle={auto_smooth_angle}"
    ]
    
    if apply_vertex_merge:
        cmd.append("--merge-verts")
    else:
        cmd.append("--no-merge-verts")
    
    # Run bpy in subprocess
    print(f"Running bpy mesh conversion: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    
    if result.returncode != 0:
        print(f"bpy stderr: {result.stderr}")
        raise RuntimeError(f"bpy mesh conversion failed: {result.stderr}")
    
    print(f"bpy stdout: {result.stdout}")
    
    if not os.path.exists(output_path):
        raise RuntimeError(f"Output file not created: {output_path}")
    
    return output_path


def load_mesh(filepath: str) -> trimesh.Trimesh:
    """Load a mesh from file."""
    mesh = trimesh.load(filepath, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)
    return mesh


def save_mesh(mesh: trimesh.Trimesh, filepath: str, file_type: str = None) -> str:
    """Save mesh to file."""
    if file_type is None:
        file_type = os.path.splitext(filepath)[1][1:].lower()
    mesh.export(filepath, file_type=file_type)
    return filepath


# Fallback functions if bpy subprocess fails (uses trimesh)
def convert_obj_to_glb_fallback(input_path: str, output_path: str) -> str:
    """Fallback GLB conversion using trimesh (no Blender features)."""
    print("Warning: Using trimesh fallback for GLB conversion")
    mesh = load_mesh(input_path)
    mesh.merge_vertices()
    mesh.fix_normals()
    mesh.export(output_path, file_type='glb')
    return output_path
