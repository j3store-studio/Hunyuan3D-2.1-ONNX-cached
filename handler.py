"""
RunPod Serverless Handler for Hunyuan3D-2.1 (Image-to-3D)

Version: 2.0.0 - Cached version (uses RunPod model caching)

Generates high-fidelity 3D models with PBR materials from input images.

API:
  Input:
    - image_base64: Base64 encoded input image (required)
    - generate_texture: Whether to generate PBR textures (default: true)
    - output_format: 'glb' or 'obj' (default: 'glb')
    - num_views: Number of views for texture (default: 6)
    - texture_resolution: Texture resolution (default: 512)

  Output:
    - model: Base64 encoded 3D model
    - format: Output format used
"""

import os
import sys
import base64
import tempfile
import traceback
from pathlib import Path


# =============================================================================
# RunPod Model Caching Support
# =============================================================================

# Possible cache locations (in order of priority)
CACHE_LOCATIONS = [
    os.environ.get("HUGGINGFACE_HUB_CACHE"),
    os.environ.get("HF_HOME"),
    "/runpod-volume/huggingface-cache/hub",
    "/runpod-volume/huggingface-cache",
]


def find_cached_model(model_name: str) -> str | None:
    """
    Find a model in RunPod's cache directory.

    Checks multiple possible cache locations:
    1. HUGGINGFACE_HUB_CACHE environment variable
    2. HF_HOME environment variable
    3. /runpod-volume/huggingface-cache/hub (RunPod default with /hub)
    4. /runpod-volume/huggingface-cache (RunPod default without /hub)

    When deploying with the Model field set in RunPod console,
    models are pre-cached at: {cache_root}/models--{org}--{name}/snapshots/{hash}/
    """
    cache_name = model_name.replace("/", "--")

    for cache_dir in CACHE_LOCATIONS:
        if not cache_dir:
            continue

        # Check both with and without /hub suffix
        dirs_to_check = [cache_dir]
        if not cache_dir.endswith("/hub"):
            dirs_to_check.append(os.path.join(cache_dir, "hub"))

        for base_dir in dirs_to_check:
            snapshots_dir = os.path.join(base_dir, f"models--{cache_name}", "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    cached_path = os.path.join(snapshots_dir, snapshots[0])
                    print(f"Found cached model at: {cached_path}")
                    return cached_path

    print(f"WARNING: No cached model found for {model_name}")
    print(f"Checked locations: {[loc for loc in CACHE_LOCATIONS if loc]}")
    return None


# Model directory: prefer cached, fallback to bundled or env var
MODEL_DIR = find_cached_model("tencent/Hunyuan3D-2.1") or os.environ.get("MODEL_DIR", "/models/Hunyuan3D-2.1")
print(f"Using model directory: {MODEL_DIR}")

# Check if model directory exists
if not os.path.exists(MODEL_DIR):
    print(f"INFO: Model not found at: {MODEL_DIR}")
    print("This is expected during build testing. Model will be loaded from RunPod cache at runtime.")
    print("Ensure the Model field is set to 'tencent/Hunyuan3D-2.1' in RunPod console.")

# Add Hunyuan3D paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/hy3dshape')
sys.path.insert(0, '/app/hy3dpaint')

import runpod
import torch


# Global pipelines (loaded once on cold start)
shape_pipeline = None
paint_pipeline = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipelines():
    """Load the shape and paint pipelines."""
    global shape_pipeline, paint_pipeline

    device = get_device()
    print(f"Using device: {device}")

    if shape_pipeline is None:
        print("Loading shape pipeline...")
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                MODEL_DIR,
                device=device
            )
            print("Shape pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading shape pipeline: {e}")
            traceback.print_exc()
            raise

    if paint_pipeline is None:
        print("Loading paint pipeline...")
        try:
            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            config = Hunyuan3DPaintConfig(
                model_path=MODEL_DIR,
                max_num_view=int(os.environ.get('MAX_NUM_VIEW', 6)),
                resolution=int(os.environ.get('TEXTURE_RESOLUTION', 512)),
                device=str(device)
            )
            paint_pipeline = Hunyuan3DPaintPipeline(config)
            print("Paint pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading paint pipeline: {e}")
            traceback.print_exc()
            raise

    return shape_pipeline, paint_pipeline


def save_base64_to_file(b64_data: str, output_path: str) -> str:
    """Decode base64 data and save to file."""
    # Handle data URI format
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]

    decoded = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(decoded)
    return output_path


def encode_file_to_base64(file_path: str) -> str:
    """Read file and encode to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Hunyuan3D-2.1.
    """
    job_input = job.get("input")

    # Health check - returns immediately without loading model
    # Used by RunPod's testing phase during deployment
    if job_input == "health_check" or (isinstance(job_input, dict) and job_input.get("health_check")):
        return {
            "status": "healthy",
            "model_dir": MODEL_DIR,
            "model_available": os.path.exists(MODEL_DIR),
            "message": "Handler ready. Model will be loaded on first inference request."
        }

    if not isinstance(job_input, dict):
        return {"error": "Invalid request: missing 'input' field"}

    # Validate required inputs
    if "image_base64" not in job_input:
        return {"error": "Missing required field: image_base64"}

    # Extract parameters
    image_b64 = job_input["image_base64"]
    generate_texture = job_input.get("generate_texture", True)
    output_format = job_input.get("output_format", "glb").lower()
    num_views = job_input.get("num_views", int(os.environ.get('MAX_NUM_VIEW', 6)))
    texture_resolution = job_input.get("texture_resolution", int(os.environ.get('TEXTURE_RESOLUTION', 512)))

    if output_format not in ["glb", "obj"]:
        return {"error": f"Invalid output_format: {output_format}. Use 'glb' or 'obj'."}

    # Load pipelines
    try:
        shape_pipe, paint_pipe = load_pipelines()
    except Exception as e:
        return {"error": f"Failed to load pipelines: {str(e)}"}

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save input image
        image_ext = ".png"
        if "image/jpeg" in image_b64[:50] or "image/jpg" in image_b64[:50]:
            image_ext = ".jpg"
        image_path = temp_path / f"input{image_ext}"

        try:
            save_base64_to_file(image_b64, str(image_path))
            print(f"Input image saved: {image_path}")
        except Exception as e:
            return {"error": f"Failed to decode input image: {str(e)}"}

        # Generate shape (untextured mesh)
        try:
            runpod.serverless.progress_update(job, "Generating 3D shape...")
            print("Generating 3D shape from image...")
            mesh = shape_pipe(image=str(image_path))[0]
            print("Shape generation complete.")
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Shape generation failed: {str(e)}"}

        # Generate texture if requested
        if generate_texture:
            try:
                runpod.serverless.progress_update(job, f"Generating textures ({num_views} views)...")
                print(f"Generating textures ({num_views} views, {texture_resolution}px)...")
                # Update paint pipeline config
                paint_pipe.config.max_num_view = num_views
                paint_pipe.config.resolution = texture_resolution
                mesh = paint_pipe(mesh, image_path=str(image_path))
                print("Texture generation complete.")
            except Exception as e:
                traceback.print_exc()
                return {"error": f"Texture generation failed: {str(e)}"}

        runpod.serverless.progress_update(job, "Exporting mesh...")
        # Export mesh
        output_path = temp_path / f"output.{output_format}"
        try:
            if output_format == "glb":
                mesh.export(str(output_path))
            else:
                mesh.export(str(output_path), file_type='obj')
            print(f"Exported to {output_format.upper()}: {output_path}")
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Failed to export mesh: {str(e)}"}

        # Encode output
        try:
            model_b64 = encode_file_to_base64(str(output_path))
            file_size_mb = os.path.getsize(str(output_path)) / (1024 * 1024)
            print(f"Output encoded: {file_size_mb:.2f} MB")
        except Exception as e:
            return {"error": f"Failed to encode output: {str(e)}"}

        return {
            "model": model_b64,
            "format": output_format,
            "textured": generate_texture
        }


# Entry point
if __name__ == "__main__":
    # Local testing: python handler.py <image.png>
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            test_image = base64.b64encode(f.read()).decode("utf-8")

        test_job = {
            "input": {
                "image_base64": test_image,
                "generate_texture": True,
                "output_format": "glb"
            }
        }
        result = handler(test_job)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success! Format: {result['format']}, Textured: {result['textured']}")
            with open("test_output.glb", "wb") as f:
                f.write(base64.b64decode(result['model']))
            print("Saved to test_output.glb")
    else:
        # Production: RunPod serverless mode
        runpod.serverless.start({"handler": handler})
