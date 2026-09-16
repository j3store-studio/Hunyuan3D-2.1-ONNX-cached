import importlib
import json
import os
import sys
import tempfile
import traceback

sys.path[:0] = ["/app/hy3dpaint", "/app/hy3dshape", "/app"]
os.chdir("/app")

failures = []


def check(name, fn):
    try:
        fn()
        print(f"ok   {name}", flush=True)
    except BaseException as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()


def versions():
    import numpy as np
    import torch

    assert torch.__version__.startswith("2.5.1"), torch.__version__
    assert np.__version__ == "1.26.4", np.__version__
    torch.from_numpy(np.zeros(3, dtype=np.float32))


def symbol(module, attr):
    def run():
        getattr(importlib.import_module(module), attr)

    return run


def realesrgan():
    from utils.image_super_utils import load_rrdbnet

    load_rrdbnet("/app/hy3dpaint/ckpt/RealESRGAN_x4plus.pth", "cpu", False)


def dino():
    from transformers import AutoImageProcessor

    dino_dir = os.environ["DINO_DIR"]
    AutoImageProcessor.from_pretrained(dino_dir)
    with open(os.path.join(dino_dir, "config.json"), encoding="utf-8") as f:
        assert json.load(f)["model_type"] == "dinov2"
    assert os.path.getsize(os.path.join(dino_dir, "model.safetensors")) > 4_000_000_000


def background_remover():
    from hy3dshape.rembg import BackgroundRemover
    from rembg import new_session

    BackgroundRemover()
    new_session("isnet-general-use")


def quality_module():
    import numpy as np
    import trimesh
    from PIL import Image

    import quality

    name, preset = quality.preset_for("ultra")
    assert name == "ultra" and preset["octree_resolution"] == 512, preset
    assert quality.preset_for("high")[1]["view_resolution"] == 768
    assert quality.preset_for("unknown")[0] == "balanced"
    view = Image.new("RGB", (256, 256), (255, 255, 255))
    view.paste((40, 40, 40), (64, 32, 192, 224))
    reference = Image.new("RGBA", (200, 300), (0, 0, 0, 0))
    reference.paste((200, 30, 30, 255), (20, 10, 180, 290))
    merged, info = quality.align_reference(reference, view, "auto")
    assert merged is not None and info["applied"], info
    pixel = np.asarray(merged)[128, 128]
    assert int(pixel[0]) > int(pixel[1]) + 40, pixel
    skipped, info = quality.align_reference(reference.rotate(90, expand=True), view, "auto")
    assert skipped is None and not info["applied"], info
    work = tempfile.mkdtemp()
    source = os.path.join(work, "white_mesh.obj")
    target = os.path.join(work, "white_mesh_remesh.obj")
    sphere = trimesh.creation.icosphere(subdivisions=5)
    sphere.export(source)
    quality.reset_job(paint_faces=5000)
    try:
        quality.remesh_keep_detail(source, target)
    finally:
        quality.reset_job()
    reduced = trimesh.load(target, force="mesh")
    assert 0 < len(reduced.faces) <= 5000, len(reduced.faces)
    floater = trimesh.creation.icosphere(subdivisions=0)
    floater.apply_translation([5.0, 0.0, 0.0])
    cleaned, info = quality.clean_mesh(trimesh.util.concatenate([sphere, floater]), 8000)
    assert info["floaters_removed"] == 20 and 0 < len(cleaned.faces) <= 8000, info


def glb_export():
    import numpy as np
    import trimesh

    import handler
    from DifferentiableRenderer.mesh_utils import load_mesh, save_mesh
    from utils.uvwrap_utils import mesh_uv_wrap

    work = tempfile.mkdtemp()
    box = mesh_uv_wrap(trimesh.creation.box())
    vtx_pos, pos_idx, vtx_uv, uv_idx, _ = load_mesh(box)
    albedo = np.full((64, 64, 3), 0.6, dtype=np.float32)
    metal_rough = np.full((64, 64, 3), 0.4, dtype=np.float32)
    obj_path = os.path.join(work, "textured_mesh.obj")
    save_mesh(obj_path, vtx_pos, pos_idx, vtx_uv, uv_idx, albedo, metallic=metal_rough, roughness=metal_rough)
    base = obj_path[:-4]
    textures = {
        "obj": obj_path,
        "albedo": base + ".jpg",
        "metallic": base + "_metallic.jpg",
        "roughness": base + "_roughness.jpg",
    }
    glb_path = os.path.join(work, "check.glb")
    handler.build_textured_glb(textures, glb_path, 32)
    with open(glb_path, "rb") as f:
        data = f.read()
    assert data[:4] == b"glTF", data[:4]
    document = json.loads(data[20 : 20 + int.from_bytes(data[12:16], "little")])
    pbr = document["materials"][0]["pbrMetallicRoughness"]
    assert "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr, pbr
    assert len(document.get("images", [])) == 2, document.get("images")


check("torch and numpy versions", versions)

for module in [
    "cv2",
    "trimesh",
    "xatlas",
    "pymeshlab",
    "skimage",
    "onnxruntime",
    "rembg",
    "timm",
    "accelerate",
    "diffusers",
    "transformers",
    "runpod",
    "custom_rasterizer",
]:
    check(f"import {module}", lambda m=module: importlib.import_module(m))

for module, attr in [
    ("DifferentiableRenderer.mesh_inpaint_processor", "meshVerticeInpaint"),
    ("DifferentiableRenderer.mesh_utils", "save_mesh"),
    ("DifferentiableRenderer.MeshRender", "MeshRender"),
    ("utils.image_super_utils", "imageSuperNet"),
    ("utils.uvwrap_utils", "mesh_uv_wrap"),
    ("utils.simplify_mesh_utils", "remesh_mesh"),
    ("utils.pipeline_utils", "ViewProcessor"),
    ("utils.multiview_utils", "multiviewDiffusionNet"),
    ("hunyuanpaintpbr.unet.modules", "Dino_v2"),
    ("hunyuanpaintpbr.pipeline", "HunyuanPaintPipeline"),
    ("textureGenPipeline", "Hunyuan3DPaintPipeline"),
    ("hy3dshape.schedulers", "FlowMatchEulerDiscreteScheduler"),
    ("hy3dshape.preprocessors", "ImageProcessorV2"),
    ("hy3dshape.models.autoencoders", "ShapeVAE"),
    ("hy3dshape.models.conditioner", "SingleImageEncoder"),
    ("hy3dshape.models.denoisers.hunyuandit", "HunYuanDiTPlain"),
    ("hy3dshape.postprocessors", "FaceReducer"),
    ("hy3dshape.pipelines", "Hunyuan3DDiTFlowMatchingPipeline"),
    ("hy3dshape.rembg", "BackgroundRemover"),
    ("quality", "install"),
    ("handler", "build_textured_glb"),
    ("handler", "parse_params"),
]:
    check(f"{module}.{attr}", symbol(module, attr))

check("RealESRGAN weights load", realesrgan)
check("dinov2-giant files", dino)
check("rembg sessions", background_remover)
check("quality presets and reference alignment", quality_module)
check("textured GLB export", glb_export)

print("=" * 60, flush=True)
if failures:
    print(f"BUILD CHECK FAILED ({len(failures)}):", flush=True)
    for item in failures:
        print(f"  - {item}", flush=True)
    sys.exit(1)
print("build check passed", flush=True)
