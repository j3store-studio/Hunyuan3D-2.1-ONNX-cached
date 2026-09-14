FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    CUDA_HOME=/usr/local/cuda \
    HF_HUB_DISABLE_TELEMETRY=1 \
    U2NET_HOME=/models/u2net \
    DINO_DIR=/models/dinov2-giant \
    HY3D_COMMIT=82920d643c0dc2f7bfd7255f45f62d386edfe60c

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        libsm6 libxext6 libxrender1 libgl1 libglib2.0-0 libgomp1 \
        build-essential gcc g++ software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel "pybind11<3"

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH}

RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app \
    && git -C /app checkout ${HY3D_COMMIT} \
    && rm -rf /app/.git

COPY requirements_worker.txt /tmp/requirements_worker.txt
RUN pip uninstall -y numpy \
    && pip install --no-cache-dir --prefer-binary -r /tmp/requirements_worker.txt \
    && python -c "import numpy, torch; assert numpy.__version__ == '1.26.4', numpy.__version__; torch.from_numpy(numpy.zeros(3)); assert torch.__version__.startswith('2.5.1'), torch.__version__"

COPY custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl \
    && rm /tmp/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl

WORKDIR /app/hy3dpaint/DifferentiableRenderer
RUN c++ -O3 -Wall -shared -std=c++17 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp \
        -o mesh_inpaint_processor$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

WORKDIR /app
RUN mkdir -p /app/hy3dpaint/ckpt \
    && wget -q -O /app/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-giant', local_dir='/models/dinov2-giant', allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])"

RUN python -c "from rembg import new_session; new_session('u2net')"

COPY schedulers.py /app/hy3dshape/hy3dshape/schedulers.py
COPY patches/mesh_utils.py /app/hy3dpaint/DifferentiableRenderer/mesh_utils.py
COPY patches/image_super_utils.py /app/hy3dpaint/utils/image_super_utils.py
COPY patches/hunyuanpaintpbr_init.py /app/hy3dpaint/hunyuanpaintpbr/__init__.py
COPY patches/build_check.py /app/build_check.py
COPY handler.py /app/handler.py

RUN python /app/build_check.py

ENV TEXTURE_SIZE=2048 \
    PREVIEW_TEXTURE_SIZE=1024 \
    MAX_FACES=40000 \
    MAX_NUM_VIEW=6 \
    VIEW_RESOLUTION=512

CMD ["python", "-u", "/app/handler.py"]
