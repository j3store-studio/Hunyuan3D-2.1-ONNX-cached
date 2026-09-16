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
DEFAULT_MAX_FACES = int(os.environ.get("MAX_FACES", "0"))
DEFAULT_VIEWS = int(os.environ.get("MAX_NUM_VIEW", "9"))
DEFAULT_VIEW_RESOLUTION = int(os.environ.get("VIEW_RESOLUTION", "512"))
FACE_CAP = int(os.environ.get("FACE_CAP", "200000"))
BACKGROUND_MODELS = {"isnet": "isnet-general-use", "u2net": None}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
MAX_BASE64_BYTES = 8 * 1024 * 1024
CONTENT_TYPES = {
    ".glb": "model/gltf-binary",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
    ".obj": "text/plain",
    ".mtl": "text/plain",
}

if not ALLOW_MODEL_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

STATE = {
    "model_dir": None,
    "shape": None,
    "paint": None,
    "rembg": None,
    "rembg_sessions": {},
    "face_reducer": None,
    "error": None,
    "load_seconds": None,
    "selftest": None,
}


class JobError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def describe_exception(exc):
    frames = traceback.extract_tb(exc.__traceback__)[-5:]
    where = " <- ".join(f"{os.path.basename(frame.filename)}:{frame.lineno} {frame.name}" for frame in reversed(frames))
    message = " ".join(str(exc).split())[:600]
    return f"{type(exc).__name__}: {message} @ {where}"


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


def gpu_selftest():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this worker")
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        "torch_arch_list": torch.cuda.get_arch_list(),
    }
    matrix = torch.randn(64, 64, device="cuda")
    report["torch_matmul"] = round(float((matrix @ matrix).abs().mean().item()), 4)

    import custom_rasterizer

    pos = torch.tensor(
        [[[-0.8, -0.8, 0.5, 1.0], [0.8, -0.8, 0.5, 1.0], [0.0, 0.8, 0.5, 1.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    findices, _ = custom_rasterizer.rasterize(pos, tri, (32, 32))
    torch.cuda.synchronize()
    report["rasterizer_pixels"] = int((findices > 0).sum().item())
    return report


def load_models():
    started = time.time()
    import torch
    import huggingface_hub

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
    import quality

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
    paint = quality.install(Hunyuan3DPaintPipeline(config))

    sessions = {}
    try:
        from rembg import new_session

        for key, name in BACKGROUND_MODELS.items():
            if not name:
                continue
            try:
                sessions[key] = new_session(name)
            except Exception as exc:
                print(f"background model {name} unavailable: {exc}", flush=True)
    except Exception as exc:
        print(f"rembg sessions unavailable: {exc}", flush=True)

    STATE.update(
        model_dir=model_dir,
        shape=shape,
        paint=paint,
        rembg=BackgroundRemover(),
        rembg_sessions=sessions,
        face_reducer=FaceReducer(),
        error=None,
        load_seconds=round(time.time() - started, 1),
    )
    print(f"models loaded from {model_dir} in {STATE['load_seconds']}s, background models {sorted(sessions)}", flush=True)


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


def first_present(inp, *keys):
    for key in keys:
        if key in inp and inp[key] is not None and inp[key] != "":
            return inp[key]
    return None


def parse_params(inp):
    import quality

    quality_name, preset = quality.preset_for(inp.get("quality"))
    background = str(inp.get("remove_background", "auto")).strip().lower()
    if background in ("1", "true", "yes"):
        background = "always"
    elif background in ("0", "false", "no"):
        background = "never"
    elif background not in ("auto", "always", "never"):
        background = "auto"
    view_resolution = clamp_int(first_present(inp, "view_resolution", "texture_resolution"), preset["view_resolution"], 512, 768)
    projection = str(first_present(inp, "reference_projection") or preset["reference_projection"]).strip().lower()
    if projection in ("1", "true", "yes", "on"):
        projection = "always"
    elif projection in ("0", "false", "no", "off"):
        projection = "never"
    elif projection not in ("auto", "always", "never"):
        projection = "auto"
    background_model = str(first_present(inp, "background_model") or preset["background_model"]).strip().lower()
    if background_model not in BACKGROUND_MODELS:
        background_model = preset["background_model"]
    output_format = str(inp.get("output_format", "glb")).strip().lower() or "glb"
    return {
        "quality": quality_name,
        "texture": as_bool(first_present(inp, "texture", "generate_texture"), True),
        "seed": clamp_int(inp.get("seed"), random.randint(0, 2**31 - 1), 0, 2**31 - 1),
        "steps": clamp_int(inp.get("steps"), preset["steps"], 5, 100),
        "guidance_scale": clamp_float(inp.get("guidance_scale"), 5.0, 1.0, 15.0),
        "octree_resolution": clamp_int(inp.get("octree_resolution"), preset["octree_resolution"], 128, 512),
        "max_faces": clamp_int(inp.get("max_faces"), min(DEFAULT_MAX_FACES or preset["max_faces"], FACE_CAP), 1000, FACE_CAP),
        "views": clamp_int(first_present(inp, "views", "num_views"), preset["views"], 6, 9),
        "view_resolution": 768 if view_resolution >= 768 else 512,
        "multiview_steps": clamp_int(inp.get("multiview_steps"), preset["multiview_steps"], 10, 50),
        "multiview_guidance": clamp_float(inp.get("multiview_guidance"), preset["multiview_guidance"], 1.0, 8.0),
        "full_texture": as_bool(inp.get("full_texture"), preset["full_texture"]),
        "jpeg_quality": clamp_int(inp.get("jpeg_quality"), preset["jpeg_quality"], 70, 100),
        "reference_projection": projection,
        "background_model": background_model,
        "remove_background": background,
        "clean_mesh": as_bool(inp.get("clean_mesh"), True),
        "output_format": "glb" if output_format not in ("glb", "obj") else output_format,
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


def remove_background(image, model):
    session = (STATE.get("rembg_sessions") or {}).get(model)
    if session is not None:
        from rembg import remove

        return remove(image.convert("RGB"), session=session, bgcolor=[255, 255, 255, 0]).convert("RGBA")
    return STATE["rembg"](image.convert("RGB")).convert("RGBA")


def prepare_subject(image, mode, model="isnet"):
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha and mode != "always":
        rgba = image.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 250:
            return rgba, False
    if mode == "never":
        return image.convert("RGBA"), False
    return remove_background(image, model), True


def load_mesh_from_url(url):
    import trimesh

    if not str(url).startswith(("https://", "http://")):
        raise JobError("bad_input", "mesh_url must be http or https")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "sudair-gen3d"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(64 * 1024 * 1024 + 1)
    except Exception as exc:
        raise JobError("bad_input", f"could not download mesh: {exc}")
    if len(data) > 64 * 1024 * 1024 or data[:4] != b"glTF":
        raise JobError("bad_input", "mesh_url is not a valid glb")
    try:
        mesh = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
    except Exception as exc:
        raise JobError("bad_input", f"could not read mesh: {exc}")
    if mesh is None or len(mesh.faces) == 0:
        raise JobError("bad_input", "mesh has no faces")
    return mesh


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
    raw_faces = int(len(mesh.faces))
    try:
        import quality

        cleaned, info = quality.clean_mesh(mesh, params["max_faces"], params["clean_mesh"])
        if len(cleaned.faces) == 0:
            raise ValueError("cleanup removed every face")
        mesh = cleaned
        params["floaters_removed"] = info["floaters_removed"]
    except Exception as exc:
        print(f"mesh cleanup fallback: {exc}", flush=True)
        if len(mesh.faces) > params["max_faces"]:
            mesh = STATE["face_reducer"](mesh, max_facenum=params["max_faces"])
    params["raw_faces"] = raw_faces
    return mesh


def generate_texture(shape_obj, subject, params, work):
    import quality

    paint = STATE["paint"]
    output_obj = os.path.join(work, "textured_mesh.obj")
    attempts = [(params["views"], params["view_resolution"])]
    if params["views"] > 6 or params["view_resolution"] > 512:
        attempts.append((6, 512))
    projection = None
    for index, (views, resolution) in enumerate(attempts):
        paint.config.max_selected_view_num = views
        paint.config.resolution = resolution
        quality.reset_job(
            steps=params["multiview_steps"],
            guidance=params["multiview_guidance"],
            seed=params["seed"],
            paint_faces=params["max_faces"],
            reference=subject if params["reference_projection"] != "never" else None,
            reference_mode=params["reference_projection"],
            downsample=not params["full_texture"],
        )
        try:
            paint(mesh_path=shape_obj, image_path=subject, output_mesh_path=output_obj, save_glb=False)
        except RuntimeError as exc:
            if index + 1 < len(attempts) and "out of memory" in str(exc).lower():
                print(f"texture stage out of memory at {views} views / {resolution}px, retrying at 6 views / 512px", flush=True)
                params["texture_fallback"] = "oom"
                release_gpu_memory()
                continue
            raise
        finally:
            projection = quality.SETTINGS.get("reference_result")
            quality.SETTINGS["reference"] = None
        params["views"] = views
        params["view_resolution"] = resolution
        break
    base = output_obj[:-4]
    textures = {
        "obj": output_obj,
        "albedo": base + ".jpg",
        "metallic": base + "_metallic.jpg",
        "roughness": base + "_roughness.jpg",
        "projection": projection,
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


def build_textured_glb(textures, out_path, max_side=None, jpeg_quality=92):
    import trimesh
    from PIL import Image

    mesh = trimesh.load(textures["obj"], force="mesh")
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise JobError("export_failed", "textured mesh has no uv coordinates")
    material_args = {
        "name": "sudair_pbr",
        "baseColorTexture": encode_jpeg(Image.open(textures["albedo"]), max_side, jpeg_quality),
    }
    if os.path.exists(textures.get("metallic", "")) and os.path.exists(textures.get("roughness", "")):
        material_args["metallicRoughnessTexture"] = encode_jpeg(
            metallic_roughness_image(textures["metallic"], textures["roughness"]), max_side, jpeg_quality
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


UPLOAD_UA = "sudair-gen3d/1.0 (+https://props.sudair.top)"


def upload_file(url, name, path):
    with open(path, "rb") as f:
        data = f.read()
    content_type = CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream")
    last_error = None
    for attempt in range(4):
        method = "POST" if attempt % 2 else "PUT"
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": content_type, "User-Agent": UPLOAD_UA, "Accept": "application/json", "X-Gen3d-Upload": name},
        )
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
                last_error = f"HTTP {response.status} ({method})"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code} ({method})"
            if exc.code < 500 and exc.code not in (403, 408, 429):
                break
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.5 * (attempt + 1))
    raise JobError("upload_failed", f"{name}: {last_error}")


def inline_fallback(files, failed):
    order = ["manifest.json", "preview.glb", "shape.glb", "model.glb", "textures/albedo.jpg"]
    results = []
    total = 0
    for name in order:
        path = files.get(name)
        if not path or name not in failed:
            continue
        size = os.path.getsize(path)
        if total + size > MAX_BASE64_BYTES:
            continue
        with open(path, "rb") as f:
            results.append(
                {
                    "name": name,
                    "b64": base64.b64encode(f.read()).decode("ascii"),
                    "bytes": size,
                    "content_type": CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream"),
                }
            )
        total += size
    return results


def run_job(job, inp, work):
    started = time.time()
    timings = {}
    params = parse_params(inp)
    put_urls = parse_put_urls(inp)
    return_base64 = as_bool(inp.get("return_base64"), not put_urls)

    progress(job, "image")
    step = time.time()
    image = load_input_image(inp)
    subject, background_removed = prepare_subject(image, params["remove_background"], params["background_model"])
    if subject.getchannel("A").getbbox() is None:
        raise JobError("empty_subject", "no object was found in the image")
    timings["image"] = round(time.time() - step, 2)

    step = time.time()
    if inp.get("mesh_url"):
        progress(job, "mesh")
        mesh = load_mesh_from_url(inp["mesh_url"])
        timings["mesh"] = round(time.time() - step, 2)
    else:
        progress(job, "shape")
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
        files["model.glb"] = build_textured_glb(textures, os.path.join(work, "model.glb"), None, params["jpeg_quality"])
        files["preview.glb"] = build_textured_glb(textures, os.path.join(work, "preview.glb"), PREVIEW_TEXTURE_SIZE, 88)
        files["textures/albedo.jpg"] = textures["albedo"]
        for key in ("metallic", "roughness"):
            if os.path.exists(textures[key]):
                files[f"textures/{key}.jpg"] = textures[key]
        if params["output_format"] == "obj":
            for path in sorted(glob.glob(os.path.join(work, "textured_mesh*"))):
                files["obj/" + os.path.basename(path)] = path
        timings["export"] = round(time.time() - step, 2)
    else:
        files["model.glb"] = shape_glb
        files["preview.glb"] = shape_glb

    texture_size = None
    if params["texture"]:
        try:
            from PIL import Image

            with Image.open(textures["albedo"]) as albedo_image:
                texture_size = int(max(albedo_image.size))
        except Exception:
            texture_size = TEXTURE_SIZE
    stats = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "raw_faces": params.get("raw_faces"),
        "floaters_removed": params.get("floaters_removed"),
        "textured": params["texture"],
        "texture_size": texture_size,
        "quality": params["quality"],
        "views": params["views"] if params["texture"] else None,
        "view_resolution": params["view_resolution"] if params["texture"] else None,
        "multiview_steps": params["multiview_steps"] if params["texture"] else None,
        "reference_projection": textures.get("projection") if params["texture"] else None,
        "texture_fallback": params.get("texture_fallback"),
        "background_removed": background_removed,
        "background_model": params["background_model"] if background_removed else None,
        "gpu_load_seconds": STATE["load_seconds"],
        "gpu": (STATE["selftest"] or {}).get("gpu"),
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
        failed = {}
        for name, path in files.items():
            if name not in put_urls:
                continue
            try:
                results.append(upload_file(put_urls[name], name, path))
            except JobError as exc:
                failed[name] = str(exc)
                print(f"upload failed for {name}: {exc}", flush=True)
        timings["upload"] = round(time.time() - step, 2)
        if failed:
            inline = inline_fallback(files, failed)
            names = {item["name"] for item in inline}
            if "preview.glb" not in names or "manifest.json" not in names:
                raise JobError("upload_failed", "; ".join(list(failed.values())[:3]))
            results.extend(inline)
            stats["upload_fallback"] = "; ".join(list(failed.values())[:3])

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
        "selftest": STATE["selftest"],
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
        text = describe_exception(exc)
        result = {"error": f"internal: {text}"}
        if "CUDA" in text or "out of memory" in text.lower():
            result["refresh_worker"] = True
        return result
    finally:
        release_gpu_memory()


def boot():
    stage = "gpu self-test"
    try:
        STATE["selftest"] = gpu_selftest()
        print(f"gpu self-test: {STATE['selftest']}", flush=True)
        stage = "model loading"
        load_models()
    except Exception as exc:
        traceback.print_exc()
        STATE["error"] = f"{stage} failed: {describe_exception(exc)}"
        print(STATE["error"], flush=True)


if __name__ == "__main__":
    import runpod

    boot()
    runpod.serverless.start({"handler": handler})
