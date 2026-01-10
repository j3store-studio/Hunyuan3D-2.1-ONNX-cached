# Hunyuan3D-2.1-ONNX-cached

RunPod serverless endpoint for Hunyuan3D-2.1 image-to-3D generation with **model caching** for faster cold starts.

## Cached Version

This is the **cached version** of Hunyuan3D-2.1-ONNX. Key differences from the original:

| Aspect | Original | Cached (this repo) |
|--------|----------|-------------------|
| Docker image size | ~50+ GB | ~15 GB |
| Cold start time | 10-20 minutes | 20-30 seconds |
| Model location | Bundled in image | RunPod cache |
| RunPod Model field | Not needed | **Required** |

**IMPORTANT:** When deploying, you MUST set the Model field to: `tencent/Hunyuan3D-2.1`

## Key Fix

This deployment uses a **prebuilt Linux wheel** for the `custom_rasterizer` CUDA extension from the official Tencent HuggingFace space, avoiding the compilation issues that occur during Docker builds.

**Prebuilt wheel source:**
```
https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/resolve/main/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl
```

**Requirements:**
- Python 3.10 (must match the wheel)
- CUDA 12.4
- PyTorch 2.5.1

## Requirements

- **GPU:** 48GB VRAM recommended (A40, A100, RTX A6000)
  - Shape generation: ~10GB VRAM
  - Texture synthesis: ~21GB VRAM
- **RunPod account** with API key

## Deployment Steps

RunPod clones your GitHub repo and builds the Dockerfile on their infrastructure.

### 1. Push to GitHub

```bash
# Create a new repo or use existing one
git init
git add Dockerfile handler.py test_input.json README.md
git commit -m "Hunyuan3D-2.1 RunPod serverless worker"
git remote add origin https://github.com/YOUR_USER/hunyuan3d-runpod.git
git push -u origin main
```

### 2. Connect GitHub to RunPod

1. Go to [RunPod Settings](https://www.runpod.io/console/settings)
2. Find **GitHub** under Connections
3. Click **Connect** and authorize RunPod

### 3. Deploy from GitHub

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Import Git Repository**
4. Choose your repository and branch (main)
5. Dockerfile path: `Dockerfile` (or leave default if in root)
6. **CRITICAL:** Set the **Model** field to: `tencent/Hunyuan3D-2.1`
   - This enables RunPod's model caching feature
   - Without this, the endpoint will fail to load the model
7. Configure endpoint:
   - **GPU:** 32-48GB (A40, A100-40GB, RTX A6000)
   - **Container Disk:** 50GB
   - **Idle Timeout:** 60s+
   - **Max Workers:** Based on budget
8. Click **Deploy Endpoint**

### 4. Monitor Build

Build stages:
- **Pending** → **Building** → **Uploading** → **Testing** → **Completed**

Check logs if build fails. Common issues:
- Dockerfile syntax errors
- Network timeouts downloading dependencies

### 5. Update Deployments

Changes to your repo don't auto-deploy. To update:
1. Create a new **GitHub Release** in your repo
2. RunPod will rebuild and redeploy

## Build Constraints (RunPod)

| Limit | Value |
|-------|-------|
| Max build time | 160 minutes |
| Max image size | 80 GB |
| GPU during build | **Not available** (why prebuilt wheel is essential) |

## Local Testing (Optional)

```bash
# Test with RunPod SDK local server
python handler.py --rp_serve_api --rp_log_level DEBUG

# Send test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

## Test the Deployed Endpoint

```bash
# Submit a job
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "'$(base64 -w0 input.png)'",
      "generate_texture": true,
      "output_format": "glb"
    }
  }'

# Check status
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_base64` | string | required | Base64 encoded input image |
| `generate_texture` | boolean | `true` | Generate PBR textures |
| `output_format` | string | `"glb"` | Output format: `"glb"` or `"obj"` |
| `num_views` | integer | `6` | Number of views for texture synthesis |
| `texture_resolution` | integer | `512` | Texture map resolution |

### Output

```json
{
  "model": "<base64 encoded GLB/OBJ file>",
  "format": "glb",
  "textured": true
}
```

### Decode the Output

```python
import base64

# From the API response
result = response.json()["output"]
model_data = base64.b64decode(result["model"])

with open(f"output.{result['format']}", "wb") as f:
    f.write(model_data)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_NUM_VIEW` | `6` | Default number of texture views |
| `TEXTURE_RESOLUTION` | `512` | Default texture resolution |
| `HF_HOME` | `/models` | HuggingFace cache directory |

## Payload Limits

RunPod has payload size limits:
- `/run` endpoint: **10 MB** maximum
- `/runsync` endpoint: **20 MB** maximum

For larger images, consider:
1. Resize images before encoding (512x512 or 1024x1024 is usually sufficient)
2. Use JPEG encoding instead of PNG for smaller payloads
3. Upload to cloud storage and pass URL instead (requires handler modification)

## Troubleshooting

### "No module named 'custom_rasterizer'"

The prebuilt wheel wasn't installed correctly. Check:
1. Python version is 3.10 (not 3.11 or 3.12)
2. The wheel URL is accessible during build
3. Run: `python -c "import custom_rasterizer; print('OK')"`

### Out of Memory

Reduce texture settings:
```json
{
  "input": {
    "image_base64": "...",
    "num_views": 4,
    "texture_resolution": 256
  }
}
```

Or disable textures entirely: `"generate_texture": false`

### Cold Starts

This cached version has significantly faster cold starts (~20-30 seconds) compared to the original (~10-20 minutes) because:

1. Model weights are pre-cached on RunPod host machines
2. Docker image is smaller (~15GB vs ~50GB)
3. No model download required at startup

To further optimize:
1. Keep Idle Timeout higher (60s+)
2. Use Active Workers to maintain minimum warm workers
3. Consider RunPod's FlashBoot feature

### Job Timeout

Default timeout is 600s. For high-resolution textures, increase via RunPod endpoint settings.

## Files

- `Dockerfile` - Container build configuration
- `handler.py` - RunPod serverless handler
- `test_input.json` - Sample test input for local testing
- `README.md` - This documentation

## Credits

- [Tencent Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Prebuilt wheel from Tencent HuggingFace Space](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1)
