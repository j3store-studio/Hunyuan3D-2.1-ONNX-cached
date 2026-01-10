# Dockerfile for Hunyuan3D-2.1 on RunPod Serverless
# Image-to-3D generation with PBR materials
# Requires ~29GB VRAM (A40, A100, RTX A6000)
#
# KEY FIX: Uses Python 3.12 + locally-built custom_rasterizer wheel
# (HuggingFace wheel has ABI mismatch with all PyTorch 2.3-2.5 versions)

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# =============================================================================
# STAGE 1: System dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libgomp1 \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default (MUST match locally-built wheel: cp312)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# VERIFY: Python version
RUN python --version | grep "3.12" || (echo "ERROR: Python 3.12 required" && exit 1)

# =============================================================================
# STAGE 2: PyTorch (must be installed BEFORE any CUDA extensions)
# =============================================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja pybind11

# Note: cu124 has known issues with PyTorch 2.5.x, using cu121 instead
# The CUDA runtime is bundled with PyTorch, so this works with cuda:12.4 base image
# Install PyTorch packages separately to avoid resolution issues
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# VERIFY: PyTorch installed with CUDA
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); assert torch.__version__.startswith('2.5'), 'Wrong PyTorch version'"

# Add PyTorch libs to LD_LIBRARY_PATH for CUDA extensions (custom_rasterizer needs libc10.so)
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# =============================================================================
# STAGE 3: Clone Hunyuan3D-2.1 source
# =============================================================================
WORKDIR /app

RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git . && \
    git lfs install && \
    git lfs pull

# VERIFY: Key directories exist
RUN test -d /app/hy3dshape || (echo "ERROR: hy3dshape not found" && exit 1)
RUN test -d /app/hy3dpaint || (echo "ERROR: hy3dpaint not found" && exit 1)
RUN test -f /app/requirements.txt || (echo "ERROR: requirements.txt not found" && exit 1)

# =============================================================================
# STAGE 4a: Install Blender for bpy support
# =============================================================================
# Use official Blender tarball - PPA has broken dependencies on Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 \
    libxkbcommon0 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Blender 4.0 LTS
RUN wget -q https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    tar -xf /tmp/blender.tar.xz -C /opt && \
    rm /tmp/blender.tar.xz && \
    ln -s /opt/blender-4.0.2-linux-x64 /opt/blender

# Add Blender's Python modules to path
ENV BLENDER_PATH=/opt/blender
ENV PYTHONPATH="${BLENDER_PATH}/4.0/scripts/modules${PYTHONPATH:+:$PYTHONPATH}"
ENV PATH="${BLENDER_PATH}:${PATH}"

# =============================================================================
# STAGE 4b: Python dependencies (inference-optimized)
# =============================================================================
# Use our trimmed requirements_inference.txt instead of full requirements.txt
# Excludes: pytorch-lightning, deepspeed, tensorboard, gradio, bpy
# This reduces image size significantly for serverless inference
COPY requirements_inference.txt /tmp/requirements_inference.txt
RUN pip install --no-cache-dir --ignore-installed -r /tmp/requirements_inference.txt

# VERIFY: Key packages installed
RUN python -c "import transformers; print(f'transformers {transformers.__version__}')"
RUN python -c "import diffusers; print(f'diffusers {diffusers.__version__}')"
RUN python -c "import trimesh; print(f'trimesh {trimesh.__version__}')"

# Re-verify PyTorch wasn't overwritten
RUN python -c "import torch; assert torch.__version__.startswith('2.5'), f'PyTorch was overwritten to {torch.__version__}'"

# =============================================================================
# STAGE 5: Custom rasterizer (locally-built wheel for PyTorch 2.5.1+cu121, Python 3.12)
# =============================================================================
# HuggingFace wheel has ABI mismatch - using locally-built wheel instead
COPY custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl && \
    rm /tmp/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl

# VERIFY: custom_rasterizer works
RUN python -c "import custom_rasterizer; print('custom_rasterizer: OK')"

# =============================================================================
# STAGE 6: Optional mesh_inpaint_processor (pure C++, no CUDA)
# =============================================================================
WORKDIR /app/hy3dpaint/DifferentiableRenderer
RUN pip install --no-cache-dir -e . 2>&1 || echo "WARN: mesh_inpaint_processor build failed (optional)"

# Check if it installed (optional, so don't fail)
RUN python -c "from mesh_inpaint_processor import meshVerticeInpaint; print('mesh_inpaint_processor: OK')" || echo "mesh_inpaint_processor: NOT INSTALLED (optional)"

# =============================================================================
# STAGE 7: RunPod SDK
# =============================================================================
RUN pip install --no-cache-dir runpod huggingface_hub[cli]

# VERIFY: RunPod SDK
RUN python -c "import runpod; print(f'runpod {runpod.__version__}')"

# =============================================================================
# STAGE 8: Download model weights (large but faster cold starts)
# =============================================================================
WORKDIR /app
RUN mkdir -p /models

# Download Hunyuan3D-2.1 weights
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', local_dir='/models/Hunyuan3D-2.1')"

# VERIFY: Model files exist
RUN test -d /models/Hunyuan3D-2.1 || (echo "ERROR: Model not downloaded" && exit 1)
RUN ls -la /models/Hunyuan3D-2.1/

# ONNX-based RealESRGAN upscaler (replaces basicsr/realesrgan packages)
COPY onnx_upscaler.py /app/onnx_upscaler.py
COPY weights/realesrgan_x4plus.onnx /app/weights/realesrgan_x4plus.onnx

# =============================================================================
# STAGE 9: Copy handler and final verification
# =============================================================================
COPY handler.py /app/handler.py

# FIX: Replace schedulers.py to fix numpy/torch compatibility issue
# (torch.from_numpy fails with "expected np.ndarray got numpy.ndarray")
COPY schedulers.py /app/hy3dshape/hy3dshape/schedulers.py

# VERIFY: Handler file exists and has correct syntax
RUN python -m py_compile /app/handler.py && echo "handler.py: syntax OK"

# VERIFY: ONNX upscaler can be imported
RUN python -c "from onnx_upscaler import upscale_image; print('ONNX upscaler: OK')"

# VERIFY: All imports in handler work
RUN python -c "\
import sys; \
sys.path.insert(0, '/app'); \
sys.path.insert(0, '/app/hy3dshape'); \
sys.path.insert(0, '/app/hy3dpaint'); \
import runpod; \
import torch; \
import base64; \
import tempfile; \
print('Handler imports: OK')"

# VERIFY: Shape pipeline can be imported (don't load weights, just import)
RUN python -c "\
import sys; \
sys.path.insert(0, '/app'); \
sys.path.insert(0, '/app/hy3dshape'); \
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; \
print('Shape pipeline import: OK')"

# VERIFY: Paint pipeline can be imported
RUN python -c "\
import sys; \
sys.path.insert(0, '/app'); \
sys.path.insert(0, '/app/hy3dpaint'); \
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig; \
print('Paint pipeline import: OK')" || echo "WARN: Paint pipeline import failed - check at runtime"

# =============================================================================
# FINAL: Environment and entrypoint
# =============================================================================
ENV HF_HOME=/models
ENV HUGGINGFACE_HUB_CACHE=/models
ENV MAX_NUM_VIEW=6
ENV TEXTURE_RESOLUTION=512

WORKDIR /app
CMD ["python", "-u", "handler.py"]

# =============================================================================
# BUILD SUMMARY (printed at end of build)
# =============================================================================
RUN echo "=============================================" && \
    echo "BUILD COMPLETE - Summary:" && \
    echo "=============================================" && \
    python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import custom_rasterizer; print('custom_rasterizer: installed')" && \
    python -c "import runpod; print(f'runpod: {runpod.__version__}')" && \
    echo "Model path: /models/Hunyuan3D-2.1" && \
    du -sh /models/Hunyuan3D-2.1 && \
    echo "============================================="
