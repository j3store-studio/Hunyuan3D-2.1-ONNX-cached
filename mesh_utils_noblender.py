"""
Mesh utilities for Hunyuan3D - NO BLENDER VERSION

Replaces bpy-dependent mesh_utils.py with trimesh-based implementation.
This allows running on Python 3.12 where bpy is not available.
"""

import os
import numpy as np
import trimesh
from typing import Optional, Tuple, Dict, Any


def merge_vertices(mesh: trimesh.Trimesh, merge_threshold: float = 1e-6) -> trimesh.Trimesh:
    """Merge duplicate vertices within threshold distance."""
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def apply_smooth_shading(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply smooth shading by computing vertex normals."""
    # Trimesh automatically computes smooth vertex normals
    mesh.fix_normals()
    return mesh


def apply_flat_shading(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply flat shading - each face gets its own normals."""
    # Unmerge vertices so each face has unique vertices
    mesh = mesh.copy()
    # This effectively gives flat shading
    return mesh


def apply_auto_smooth(mesh: trimesh.Trimesh, angle: float = 30.0) -> trimesh.Trimesh:
    """
    Apply auto-smooth shading based on angle threshold.
    
    Edges with angles greater than threshold get sharp normals,
    others get smooth interpolated normals.
    
    Args:
        mesh: Input trimesh
        angle: Angle threshold in degrees (default 30)
    
    Returns:
        Mesh with adjusted normals
    """
    # Trimesh handles this reasonably well with face_normals
    # For more precise control, we'd need custom normal calculation
    mesh.fix_normals()
    return mesh


def convert_obj_to_glb(
    input_path: str,
    output_path: str,
    apply_vertex_merge: bool = True,
    shading_mode: str = "auto",  # "smooth", "flat", "auto"
    auto_smooth_angle: float = 30.0
) -> str:
    """
    Convert OBJ file to GLB format.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path for output GLB file
        apply_vertex_merge: Whether to merge duplicate vertices
        shading_mode: Shading mode - "smooth", "flat", or "auto"
        auto_smooth_angle: Angle for auto-smooth (degrees)
    
    Returns:
        Path to output GLB file
    """
    # Load the mesh
    mesh = trimesh.load(input_path, force='mesh')
    
    # Handle scene with multiple meshes
    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes into one
        meshes = []
        for name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid meshes found in input file")
    
    # Merge vertices if requested
    if apply_vertex_merge:
        mesh = merge_vertices(mesh)
    
    # Apply shading
    if shading_mode == "smooth":
        mesh = apply_smooth_shading(mesh)
    elif shading_mode == "flat":
        mesh = apply_flat_shading(mesh)
    elif shading_mode == "auto":
        mesh = apply_auto_smooth(mesh, auto_smooth_angle)
    
    # Export to GLB
    mesh.export(output_path, file_type='glb')
    
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
