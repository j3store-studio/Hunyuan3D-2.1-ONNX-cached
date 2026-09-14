import json
import os
import sys
import tempfile

sys.path[:0] = ["/app/hy3dpaint", "/app/hy3dshape", "/app"]
os.chdir("/app")

import numpy as np
import torch
import trimesh

assert torch.__version__.startswith("2.5.1"), torch.__version__
assert np.__version__ == "1.26.4", np.__version__
torch.from_numpy(np.zeros(3, dtype=np.float32))

import custom_rasterizer
from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint
from DifferentiableRenderer.mesh_utils import load_mesh, save_mesh
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.postprocessors import FaceReducer
from hy3dshape.rembg import BackgroundRemover
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
from transformers import AutoImageProcessor
from utils.image_super_utils import load_rrdbnet
from utils.uvwrap_utils import mesh_uv_wrap

import handler

load_rrdbnet("/app/hy3dpaint/ckpt/RealESRGAN_x4plus.pth", "cpu", False)

dino_dir = os.environ["DINO_DIR"]
AutoImageProcessor.from_pretrained(dino_dir)
with open(os.path.join(dino_dir, "config.json"), encoding="utf-8") as f:
    assert json.load(f)["model_type"] == "dinov2"
assert os.path.getsize(os.path.join(dino_dir, "model.safetensors")) > 4_000_000_000

BackgroundRemover()

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

print("build check passed")
