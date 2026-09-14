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

    BackgroundRemover()


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
    ("handler", "build_textured_glb"),
]:
    check(f"{module}.{attr}", symbol(module, attr))

check("RealESRGAN weights load", realesrgan)
check("dinov2-giant files", dino)
check("rembg u2net session", background_remover)
check("textured GLB export", glb_export)

print("=" * 60, flush=True)
if failures:
    print(f"BUILD CHECK FAILED ({len(failures)}):", flush=True)
    for item in failures:
        print(f"  - {item}", flush=True)
    sys.exit(1)
print("build check passed", flush=True)
