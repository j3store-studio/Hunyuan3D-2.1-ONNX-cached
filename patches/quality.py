import math

import numpy as np
from PIL import Image, ImageFilter

PRESETS = {
    "balanced": {
        "steps": 50,
        "octree_resolution": 384,
        "max_faces": 80000,
        "views": 6,
        "view_resolution": 512,
        "multiview_steps": 20,
        "multiview_guidance": 3.0,
        "full_texture": False,
        "jpeg_quality": 92,
        "reference_projection": "auto",
        "background_model": "isnet",
    },
    "high": {
        "steps": 50,
        "octree_resolution": 384,
        "max_faces": 120000,
        "views": 7,
        "view_resolution": 768,
        "multiview_steps": 25,
        "multiview_guidance": 3.0,
        "full_texture": True,
        "jpeg_quality": 94,
        "reference_projection": "auto",
        "background_model": "isnet",
    },
    "ultra": {
        "steps": 50,
        "octree_resolution": 512,
        "max_faces": 160000,
        "views": 9,
        "view_resolution": 768,
        "multiview_steps": 30,
        "multiview_guidance": 3.0,
        "full_texture": True,
        "jpeg_quality": 95,
        "reference_projection": "auto",
        "background_model": "isnet",
    },
}

PRESET_ALIASES = {
    "normal": "balanced",
    "standard": "balanced",
    "default": "balanced",
    "medium": "balanced",
    "fast": "balanced",
    "max": "ultra",
    "best": "ultra",
}

SETTINGS = {
    "steps": None,
    "guidance": None,
    "seed": None,
    "paint_faces": 40000,
    "reference": None,
    "reference_used": False,
    "reference_mode": "auto",
    "reference_result": None,
    "downsample": True,
    "front_mask": None,
}


def preset_for(name):
    key = str(name or "balanced").strip().lower()
    key = PRESET_ALIASES.get(key, key)
    if key not in PRESETS:
        key = "balanced"
    return key, dict(PRESETS[key])


def reset_job(**values):
    SETTINGS.update(
        steps=None,
        guidance=None,
        seed=None,
        paint_faces=40000,
        reference=None,
        reference_used=False,
        reference_mode="auto",
        reference_result=None,
        downsample=True,
        front_mask=None,
    )
    SETTINGS.update(values)


def is_front(elev, azim):
    return int(round(float(elev))) == 0 and int(round(float(azim))) % 360 == 0


class MultiviewProxy:
    def __init__(self, pipeline, settings):
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_settings", settings)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_pipeline"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_pipeline"), name, value)

    def __call__(self, *args, **kwargs):
        pipeline = object.__getattribute__(self, "_pipeline")
        settings = object.__getattribute__(self, "_settings")
        steps = settings.get("steps")
        if steps:
            kwargs["num_inference_steps"] = int(steps)
        guidance = settings.get("guidance")
        if guidance is not None:
            kwargs["guidance_scale"] = float(guidance)
        seed = settings.get("seed")
        if seed is not None:
            import torch

            kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(int(seed))
        return pipeline(*args, **kwargs)


def decimate_to(mesh, limit):
    import pymeshlab
    import trimesh

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.ascontiguousarray(mesh.faces, dtype=np.int32),
        )
    )
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=int(limit),
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
        autoclean=True,
    )
    current = ms.current_mesh()
    return trimesh.Trimesh(vertices=current.vertex_matrix(), faces=current.face_matrix(), process=False)


def clean_mesh(mesh, limit, remove_floaters=True, floater_ratio=0.005, max_removed=0.1):
    import pymeshlab
    import trimesh

    faces_in = int(len(mesh.faces))
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.ascontiguousarray(mesh.faces, dtype=np.int32),
        )
    )
    info = {"faces_in": faces_in, "floaters_removed": 0}
    if remove_floaters and faces_in > 0:
        ms.apply_filter("compute_selection_by_small_disconnected_components_per_face", nbfaceratio=float(floater_ratio))
        selected = int(ms.current_mesh().selected_face_number())
        if 0 < selected <= faces_in * max_removed:
            ms.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
            ms.apply_filter("meshing_remove_selected_vertices_and_faces")
            info["floaters_removed"] = selected
        elif selected:
            ms.apply_filter("set_selection_none", allfaces=True, allverts=True)
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    if ms.current_mesh().face_number() > int(limit):
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=int(limit),
            qualitythr=1.0,
            preserveboundary=True,
            boundaryweight=3,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
            autoclean=True,
        )
    current = ms.current_mesh()
    cleaned = trimesh.Trimesh(vertices=current.vertex_matrix(), faces=current.face_matrix(), process=False)
    info["faces_out"] = int(len(cleaned.faces))
    return cleaned, info


def remesh_keep_detail(mesh_path, remesh_path, *args, **kwargs):
    import trimesh

    limit = int(SETTINGS.get("paint_faces") or 40000)
    mesh = trimesh.load(mesh_path, force="mesh")
    if len(mesh.faces) > limit:
        mesh = decimate_to(mesh, limit)
    mesh.export(remesh_path)
    return remesh_path


def foreground_box(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 8 or len(cols) < 8:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def erode(mask, n):
    m = mask.copy()
    for _ in range(n):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def view_foreground(view, front_mask=None):
    if front_mask is not None:
        mask = np.asarray(front_mask, dtype=bool)
        if mask.shape != (view.size[1], view.size[0]):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(view.size, Image.NEAREST)) > 127
        return erode(mask, 1)
    pixels = np.asarray(view.convert("RGB"), dtype=np.float32) / 255.0
    return erode(pixels.min(axis=2) < 0.94, 2)


def match_colors(ref_rgb, view_rgb, region):
    if int(region.sum()) < 64:
        return ref_rgb, [1.0, 1.0, 1.0]
    ref_mean = ref_rgb[region].reshape(-1, 3).mean(axis=0)
    view_mean = view_rgb[region].reshape(-1, 3).mean(axis=0)
    gain = np.clip(view_mean / np.maximum(ref_mean, 1.0), 0.6, 1.6)
    return np.clip(ref_rgb * gain, 0, 255), [round(float(g), 3) for g in gain]


def align_reference(reference, view, mode="auto", front_mask=None, min_iou=0.72, max_aspect_diff=0.25):
    view = view.convert("RGB")
    view_mask = view_foreground(view, front_mask)
    vbox = foreground_box(view_mask)
    ref = reference.convert("RGBA")
    alpha = np.asarray(ref.getchannel("A")) > 127
    rbox = foreground_box(alpha)
    if vbox is None or rbox is None:
        return None, {"applied": False, "reason": "no_foreground", "mode": mode}
    vw, vh = vbox[2] - vbox[0], vbox[3] - vbox[1]
    rw, rh = rbox[2] - rbox[0], rbox[3] - rbox[1]
    aspect_diff = abs(math.log((vw / max(vh, 1)) / (rw / max(rh, 1))))
    ref_crop = ref.crop(rbox).resize((vw, vh), Image.LANCZOS)
    ref_alpha = np.asarray(ref_crop.getchannel("A")) > 127
    view_crop = view_mask[vbox[1] : vbox[3], vbox[0] : vbox[2]]
    union = float((ref_alpha | view_crop).sum())
    iou = float((ref_alpha & view_crop).sum()) / max(union, 1.0)
    info = {
        "iou": round(iou, 3),
        "aspect_diff": round(aspect_diff, 3),
        "mode": mode,
        "mask": "geometry" if front_mask is not None else "color",
    }
    if mode == "never" or (mode != "always" and (iou < min_iou or aspect_diff > max_aspect_diff)):
        info["applied"] = False
        return None, info
    view_np = np.asarray(view, dtype=np.float32)[vbox[1] : vbox[3], vbox[0] : vbox[2]]
    ref_rgb = np.asarray(ref_crop.convert("RGB"), dtype=np.float32)
    ref_rgb, gain = match_colors(ref_rgb, view_np, ref_alpha & view_crop)
    patch = Image.fromarray(ref_rgb.round().astype(np.uint8), "RGB")
    soft = ref_crop.getchannel("A").filter(ImageFilter.GaussianBlur(1.2))
    canvas = view.copy()
    canvas.paste(patch, (vbox[0], vbox[1]), soft)
    info["applied"] = True
    info["gain"] = gain
    return canvas, info


def install(paint, settings=None):
    settings = SETTINGS if settings is None else settings
    try:
        import textureGenPipeline

        textureGenPipeline.remesh_mesh = remesh_keep_detail
    except Exception:
        pass
    models = getattr(paint, "models", None) or {}
    multiview = models.get("multiview_model")
    if multiview is not None and not isinstance(getattr(multiview, "pipeline", None), MultiviewProxy):
        multiview.pipeline = MultiviewProxy(multiview.pipeline, settings)
    processor = paint.view_processor
    if not getattr(processor, "quality_installed", False):
        original_normals = processor.render_normal_multiview
        original_bake = processor.bake_from_multiview

        def render_normal_multiview(camera_elevs, camera_azims, use_abs_coor=True):
            maps = original_normals(camera_elevs, camera_azims, use_abs_coor=use_abs_coor)
            settings["front_mask"] = None
            for index, (elev, azim) in enumerate(zip(camera_elevs, camera_azims)):
                if is_front(elev, azim) and index < len(maps):
                    settings["front_mask"] = np.asarray(maps[index].convert("RGB")).min(axis=2) < 250
                    break
            return maps

        def bake_from_multiview(views, camera_elevs, camera_azims, view_weights):
            reference = settings.get("reference")
            if reference is not None and not settings.get("reference_used"):
                settings["reference_used"] = True
                for index, (elev, azim) in enumerate(zip(camera_elevs, camera_azims)):
                    if is_front(elev, azim) and index < len(views):
                        try:
                            new_view, info = align_reference(
                                reference, views[index], settings.get("reference_mode", "auto"), settings.get("front_mask")
                            )
                        except Exception as exc:
                            new_view, info = None, {"applied": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
                        settings["reference_result"] = info
                        if new_view is not None:
                            views = list(views)
                            views[index] = new_view
                        break
            return original_bake(views, camera_elevs, camera_azims, view_weights)

        processor.render_normal_multiview = render_normal_multiview
        processor.bake_from_multiview = bake_from_multiview
        processor.quality_installed = True
    render = paint.render
    if not getattr(render, "quality_installed", False):
        original_save = render.save_mesh

        def save_mesh(mesh_path, downsample=False):
            return original_save(mesh_path, downsample=bool(settings.get("downsample", downsample)))

        render.save_mesh = save_mesh
        render.quality_installed = True
    return paint
