import base64
import gc
import glob
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request

APP_DIR = "/app"
sys.path[:0] = [f"{APP_DIR}/hy3dpaint", f"{APP_DIR}/hy3dshape", APP_DIR]
os.chdir(APP_DIR)

MODEL_ID = "tencent/Hunyuan3D-2.1"
SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"
PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"
DINO_DIR = os.environ.get("DINO_DIR", "/models/dinov2-giant")
REALESRGAN_PATH = f"{APP_DIR}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
ALLOW_MODEL_DOWNLOAD = os.environ.get("ALLOW_MODEL_DOWNLOAD", "0") == "1"
TEXTURE_SIZE = int(os.environ.get("TEXTURE_SIZE", "2048"))
PREVIEW_TEXTURE_SIZE = int(os.environ.get("PREVIEW_TEXTURE_SIZE", "1024"))
DEFAULT_MAX_FACES = int(os.environ.get("MAX_FACES", "40000"))
DEFAULT_VIEWS = int(os.environ.get("MAX_NUM_VIEW", "6"))
DEFAULT_VIEW_RESOLUTION = int(os.environ.get("VIEW_RESOLUTION", "512"))
FACE_CAP = 40000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
MAX_BASE64_BYTES = 8 * 1024 * 1024
CONTENT_TYPES = {".glb": "model/gltf-binary", ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json"}

if not ALLOW_MODEL_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

STATE = {
    "model_dir": None,
    "shape": None,
    "paint": None,
    "rembg": None,
    "face_reducer": None,
    "error": None,
    "load_seconds": None,
}


class JobError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def find_cached_model():
    override = os.environ.get("MODEL_DIR")
    if override and os.path.isdir(os.path.join(override, SHAPE_SUBFOLDER)):
        return override
    roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "/runpod-volume/huggingface-cache/hub",
        "/runpod-volume/huggingface-cache",
    ]
    repo_dir = "models--" + MODEL_ID.replace("/", "--")
    for root in [r for r in roots if r]:
        base = os.path.join(root, repo_dir)
        candidates = []
        ref = os.path.join(base, "refs", "main")
        if os.path.isfile(ref):
            with open(ref, encoding="utf-8") as f:
                candidates.append(os.path.join(base, "snapshots", f.read().strip()))
        candidates += sorted(glob.glob(os.path.join(base, "snapshots", "*")), key=os.path.getmtime, reverse=True)
        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, SHAPE_SUBFOLDER, "model.fp16.ckpt")) and os.path.isdir(
                os.path.join(candidate, PAINT_SUBFOLDER)
            ):
                return candidate
    return None


def resolve_model_dir():
    found = find_cached_model()
    if found or not ALLOW_MODEL_DOWNLOAD:
        return found
    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        allow_patterns=[f"{SHAPE_SUBFOLDER}/*", f"{PAINT_SUBFOLDER}/*"],
        local_dir="/models/Hunyuan3D-2.1",
    )


def load_models():
    started = time.time()
    import torch
    import huggingface_hub

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this worker")
    model_dir = resolve_model_dir()
    if not model_dir:
        raise RuntimeError("cached model not found, set the endpoint Model field to tencent/Hunyuan3D-2.1")

    real_snapshot_download = huggingface_hub.snapshot_download

    def local_snapshot_download(repo_id=None, *args, **kwargs):
        if repo_id == MODEL_ID:
            return model_dir
        return real_snapshot_download(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = local_snapshot_download
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import FaceReducer
    from hy3dshape.rembg import BackgroundRemover
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_dir,
        subfolder=SHAPE_SUBFOLDER,
        use_safetensors=False,
        device="cuda",
    )

    config = Hunyuan3DPaintConfig(DEFAULT_VIEWS, DEFAULT_VIEW_RESOLUTION)
    config.multiview_pretrained_path = MODEL_ID
    config.multiview_cfg_path = f"{APP_DIR}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.custom_pipeline = f"{APP_DIR}/hy3dpaint/hunyuanpaintpbr"
    config.dino_ckpt_path = DINO_DIR
    config.realesrgan_ckpt_path = REALESRGAN_PATH
    config.texture_size = TEXTURE_SIZE * 2
    paint = Hunyuan3DPaintPipeline(config)

    STATE.update(
        model_dir=model_dir,
        shape=shape,
        paint=paint,
        rembg=BackgroundRemover(),
        face_reducer=FaceReducer(),
        error=None,
        load_seconds=round(time.time() - started, 1),
    )
    print(f"models loaded from {model_dir} in {STATE['load_seconds']}s", flush=True)


def progress(job, message):
    try:
        import runpod

        runpod.serverless.progress_update(job, message)
    except Exception:
        pass


def clamp_int(value, default, low, high):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def clamp_float(value, default, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_params(inp):
    background = str(inp.get("remove_background", "auto")).strip().lower()
    if background in ("1", "true", "yes"):
        background = "always"
    elif background in ("0", "false", "no"):
        background = "never"
    elif background not in ("auto", "always", "never"):
        background = "auto"
    view_resolution = clamp_int(inp.get("view_resolution"), DEFAULT_VIEW_RESOLUTION, 512, 768)
    return {
        "texture": as_bool(inp.get("texture", inp.get("generate_texture")), True),
        "seed": clamp_int(inp.get("seed"), random.randint(0, 2**31 - 1), 0, 2**31 - 1),
        "steps": clamp_int(inp.get("steps"), 50, 5, 100),
        "guidance_scale": clamp_float(inp.get("guidance_scale"), 5.0, 1.0, 15.0),
        "octree_resolution": clamp_int(inp.get("octree_resolution"), 256, 128, 512),
        "max_faces": clamp_int(inp.get("max_faces"), min(DEFAULT_MAX_FACES, FACE_CAP), 1000, FACE_CAP),
        "views": clamp_int(inp.get("views"), DEFAULT_VIEWS, 6, 9),
        "view_resolution": 768 if view_resolution >= 768 else 512,
        "remove_background": background,
    }


def parse_put_urls(inp):
    output = inp.get("output") or {}
    if not isinstance(output, dict):
        raise JobError("bad_input", "output must be an object")
    put_urls = output.get("put_urls") or {}
    if not isinstance(put_urls, dict):
        raise JobError("bad_input", "output.put_urls must be an object")
    for name, url in put_urls.items():
        if not isinstance(url, str) or not url.startswith("https://"):
            raise JobError("bad_input", f"put url for {name} must be https")
    if put_urls and "model.glb" not in put_urls:
        raise JobError("bad_input", "output.put_urls must include model.glb")
    return put_urls


def load_input_image(inp):
    from PIL import Image, ImageOps

    url = inp.get("image_url")
    raw = inp.get("image_base64")
    try:
        if url:
            if not str(url).startswith(("https://", "http://")):
                raise JobError("bad_image", "image_url must be http or https")
            request = urllib.request.Request(url, headers={"User-Agent": "sudair-gen3d"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
        elif raw:
            if raw.startswith("data:") and "," in raw[:200]:
                raw = raw.split(",", 1)[1]
            data = base64.b64decode(raw)
        else:
            raise JobError("bad_input", "image_url or image_base64 is required")
    except JobError:
        raise
    except Exception as exc:
        raise JobError("bad_image", f"could not read the image: {exc}")
    if len(data) > MAX_IMAGE_BYTES:
        raise JobError("bad_image", "image is larger than 20MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise JobError("bad_image", f"not a valid image: {exc}")
    image = ImageOps.exif_transpose(image)
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image


def prepare_subject(image, mode):
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha and mode != "always":
        rgba = image.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 250:
            return rgba, False
    if mode == "never":
        return image.convert("RGBA"), False
    return STATE["rembg"](image.convert("RGB")).convert("RGBA"), True


def generate_shape(subject, params):
    import torch

    generator = torch.Generator().manual_seed(params["seed"])
    meshes = STATE["shape"](
        image=subject,
        num_inference_steps=params["steps"],
        guidance_scale=params["guidance_scale"],
        generator=generator,
        octree_resolution=params["octree_resolution"],
        num_chunks=200000,
        output_type="trimesh",
    )
    mesh = meshes[0] if meshes else None
    if mesh is None or len(mesh.faces) == 0:
        raise JobError("shape_failed", "could not extract a surface from this image")
    return STATE["face_reducer"](mesh, max_facenum=params["max_faces"])


def generate_texture(shape_obj, subject, params, work):
    paint = STATE["paint"]
    paint.config.max_selected_view_num = params["views"]
    paint.config.resolution = params["view_resolution"]
    output_obj = os.path.join(work, "textured_mesh.obj")
    paint(mesh_path=shape_obj, image_path=subject, output_mesh_path=output_obj, save_glb=False)
    base = output_obj[:-4]
    textures = {
        "obj": output_obj,
        "albedo": base + ".jpg",
        "metallic": base + "_metallic.jpg",
        "roughness": base + "_roughness.jpg",
    }
    if not os.path.exists(output_obj) or not os.path.exists(textures["albedo"]):
        raise JobError("texture_failed", "texture pipeline did not produce a textured mesh")
    return textures


def encode_jpeg(image, max_side=None, quality=92):
    from PIL import Image

    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    result = Image.open(buffer)
    result.load()
    return result


def metallic_roughness_image(metallic_path, roughness_path):
    import numpy as np
    from PIL import Image

    metallic = Image.open(metallic_path).convert("L")
    roughness = Image.open(roughness_path).convert("L")
    if roughness.size != metallic.size:
        roughness = roughness.resize(metallic.size, Image.Resampling.BILINEAR)
    packed = np.zeros((metallic.size[1], metallic.size[0], 3), dtype=np.uint8)
    packed[..., 0] = 255
    packed[..., 1] = np.asarray(roughness)
    packed[..., 2] = np.asarray(metallic)
    return Image.fromarray(packed, "RGB")


def build_textured_glb(textures, out_path, max_side=None):
    import trimesh
    from PIL import Image

    mesh = trimesh.load(textures["obj"], force="mesh")
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise JobError("export_failed", "textured mesh has no uv coordinates")
    material_args = {
        "name": "sudair_pbr",
        "baseColorTexture": encode_jpeg(Image.open(textures["albedo"]), max_side),
    }
    if os.path.exists(textures.get("metallic", "")) and os.path.exists(textures.get("roughness", "")):
        material_args["metallicRoughnessTexture"] = encode_jpeg(
            metallic_roughness_image(textures["metallic"], textures["roughness"]), max_side
        )
        material_args["metallicFactor"] = 1.0
        material_args["roughnessFactor"] = 1.0
    else:
        material_args["metallicFactor"] = 0.0
        material_args["roughnessFactor"] = 1.0
    material = trimesh.visual.material.PBRMaterial(**material_args)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.export(out_path, file_type="glb", include_normals=True)
    return out_path


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_file(url, name, path):
    with open(path, "rb") as f:
        data = f.read()
    content_type = CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream")
    last_error = None
    for attempt in range(4):
        request = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if 200 <= response.status < 300:
                    return {
                        "name": name,
                        "stored": True,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "content_type": content_type,
                    }
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code < 500 and exc.code != 429:
                break
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2 * (attempt + 1))
    raise JobError("upload_failed", f"{name}: {last_error}")


def run_job(job, inp, work):
    started = time.time()
    timings = {}
    params = parse_params(inp)
    put_urls = parse_put_urls(inp)
    return_base64 = as_bool(inp.get("return_base64"), not put_urls)

    progress(job, "image")
    step = time.time()
    image = load_input_image(inp)
    subject, background_removed = prepare_subject(image, params["remove_background"])
    if subject.getchannel("A").getbbox() is None:
        raise JobError("empty_subject", "no object was found in the image")
    timings["image"] = round(time.time() - step, 2)

    progress(job, "shape")
    step = time.time()
    mesh = generate_shape(subject, params)
    shape_obj = os.path.join(work, "white_mesh.obj")
    shape_glb = os.path.join(work, "shape.glb")
    mesh.export(shape_obj)
    mesh.export(shape_glb, file_type="glb", include_normals=True)
    timings["shape"] = round(time.time() - step, 2)

    files = {"shape.glb": shape_glb}
    if params["texture"]:
        progress(job, "texture")
        step = time.time()
        textures = generate_texture(shape_obj, subject, params, work)
        timings["texture"] = round(time.time() - step, 2)

        progress(job, "export")
        step = time.time()
        files["model.glb"] = build_textured_glb(textures, os.path.join(work, "model.glb"))
        files["preview.glb"] = build_textured_glb(textures, os.path.join(work, "preview.glb"), PREVIEW_TEXTURE_SIZE)
        files["textures/albedo.jpg"] = textures["albedo"]
        for key in ("metallic", "roughness"):
            if os.path.exists(textures[key]):
                files[f"textures/{key}.jpg"] = textures[key]
        timings["export"] = round(time.time() - step, 2)
    else:
        files["model.glb"] = shape_glb
        files["preview.glb"] = shape_glb

    stats = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "textured": params["texture"],
        "texture_size": TEXTURE_SIZE if params["texture"] else None,
        "background_removed": background_removed,
        "gpu_load_seconds": STATE["load_seconds"],
    }

    manifest_path = os.path.join(work, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "contract": 1,
                "files": {name: {"bytes": os.path.getsize(path), "sha256": sha256_of(path)} for name, path in files.items()},
                "stats": stats,
                "params": params,
                "timings": timings,
            },
            f,
        )
    files["manifest.json"] = manifest_path

    results = []
    if put_urls:
        progress(job, "upload")
        step = time.time()
        for name, path in files.items():
            if name in put_urls:
                results.append(upload_file(put_urls[name], name, path))
        timings["upload"] = round(time.time() - step, 2)

    if return_base64:
        path = files["preview.glb"]
        size = os.path.getsize(path)
        if size > MAX_BASE64_BYTES:
            raise JobError("too_large", f"preview.glb is {size} bytes, send output.put_urls instead")
        with open(path, "rb") as f:
            results.append(
                {
                    "name": "preview.glb",
                    "b64": base64.b64encode(f.read()).decode("ascii"),
                    "bytes": size,
                    "content_type": "model/gltf-binary",
                }
            )

    timings["total"] = round(time.time() - started, 2)
    return {"contract": 1, "files": results, "stats": stats, "seed": params["seed"], "timings": timings}


def health():
    import torch

    cuda = torch.cuda.is_available()
    return {
        "status": "ready" if STATE["shape"] is not None else "not_ready",
        "error": STATE["error"],
        "model_dir": STATE["model_dir"],
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "load_seconds": STATE["load_seconds"],
    }


def release_gpu_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def handler(job):
    inp = job.get("input")
    if inp == "health_check" or (isinstance(inp, dict) and inp.get("health_check")):
        return health()
    if not isinstance(inp, dict):
        return {"error": "bad_input: input must be an object"}
    if STATE["shape"] is None:
        return {"error": f"not_ready: {STATE['error'] or 'models are not loaded'}"}
    try:
        with tempfile.TemporaryDirectory(prefix="gen3d_") as work:
            return run_job(job, inp, work)
    except JobError as exc:
        return {"error": f"{exc.code}: {exc}"}
    except Exception as exc:
        traceback.print_exc()
        text = f"{type(exc).__name__}: {exc}"
        result = {"error": f"internal: {text}"}
        if "CUDA" in text or "out of memory" in text.lower():
            result["refresh_worker"] = True
        return result
    finally:
        release_gpu_memory()


def boot():
    try:
        load_models()
    except Exception as exc:
        traceback.print_exc()
        STATE["error"] = f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    import runpod

    boot()
    runpod.serverless.start({"handler": handler})
