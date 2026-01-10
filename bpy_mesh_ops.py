#!/opt/bpy-env/bin/python
"""
Blender Python (bpy) mesh operations wrapper.

This script runs in the Python 3.10 bpy environment and is called
via subprocess from the main Python 3.12 handler.

Usage:
    /opt/bpy-env/bin/python bpy_mesh_ops.py convert_obj_to_glb input.obj output.glb [options]
    
Commands:
    convert_obj_to_glb <input> <output> [--merge-verts] [--shading=auto|smooth|flat] [--angle=30]
"""

import sys
import os
import argparse
import bpy
import math


def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath):
    """Import OBJ file into Blender."""
    bpy.ops.wm.obj_import(filepath=filepath)


def export_glb(filepath):
    """Export scene to GLB format."""
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=False
    )


def select_mesh_objects():
    """Select all mesh objects in scene."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj


def merge_vertices(threshold=0.0001):
    """Merge vertices by distance."""
    select_mesh_objects()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_smooth_shading():
    """Apply smooth shading to all mesh objects."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for poly in obj.data.polygons:
                poly.use_smooth = True


def apply_flat_shading():
    """Apply flat shading to all mesh objects."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for poly in obj.data.polygons:
                poly.use_smooth = False


def apply_auto_smooth(angle_degrees=30.0):
    """Apply auto-smooth shading based on angle."""
    angle_radians = math.radians(angle_degrees)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Blender 4.0+ uses different API for auto-smooth
            if hasattr(obj.data, 'use_auto_smooth'):
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = angle_radians
            else:
                # Blender 4.1+ - auto smooth is per-modifier or via shade_smooth_by_angle
                try:
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    bpy.ops.object.shade_smooth_by_angle(angle=angle_radians)
                except:
                    # Fallback to smooth shading
                    for poly in obj.data.polygons:
                        poly.use_smooth = True


def convert_obj_to_glb(input_path, output_path, merge_verts=True, shading='auto', angle=30.0):
    """
    Convert OBJ to GLB with optional processing.
    
    Args:
        input_path: Input OBJ file
        output_path: Output GLB file
        merge_verts: Whether to merge duplicate vertices
        shading: 'smooth', 'flat', or 'auto'
        angle: Auto-smooth angle in degrees
    """
    # Start fresh
    clear_scene()
    
    # Import OBJ
    print(f"Importing: {input_path}")
    import_obj(input_path)
    
    # Merge vertices if requested
    if merge_verts:
        print("Merging vertices...")
        merge_vertices()
    
    # Apply shading
    print(f"Applying {shading} shading...")
    if shading == 'smooth':
        apply_smooth_shading()
    elif shading == 'flat':
        apply_flat_shading()
    elif shading == 'auto':
        apply_auto_smooth(angle)
    
    # Export GLB
    print(f"Exporting: {output_path}")
    export_glb(output_path)
    
    print("Done!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Blender mesh operations')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # convert_obj_to_glb command
    convert_parser = subparsers.add_parser('convert_obj_to_glb', help='Convert OBJ to GLB')
    convert_parser.add_argument('input', help='Input OBJ file')
    convert_parser.add_argument('output', help='Output GLB file')
    convert_parser.add_argument('--merge-verts', action='store_true', default=True, help='Merge duplicate vertices')
    convert_parser.add_argument('--no-merge-verts', action='store_false', dest='merge_verts', help='Do not merge vertices')
    convert_parser.add_argument('--shading', choices=['smooth', 'flat', 'auto'], default='auto', help='Shading mode')
    convert_parser.add_argument('--angle', type=float, default=30.0, help='Auto-smooth angle in degrees')
    
    args = parser.parse_args()
    
    if args.command == 'convert_obj_to_glb':
        success = convert_obj_to_glb(
            args.input,
            args.output,
            merge_verts=args.merge_verts,
            shading=args.shading,
            angle=args.angle
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
