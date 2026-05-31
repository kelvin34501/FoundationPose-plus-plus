# Local Installation Without Docker

This document records the environment inferred from
`shingarey/foundationpose_custom_cuda121:latest` and gives a conda-based setup
for machines that cannot run Docker.

## Docker Image Summary

The Docker Hub image does not publish a source Dockerfile. Its image history
shows the following environment:

- Image: `shingarey/foundationpose_custom_cuda121:latest`
- Last updated: `2024-04-09`
- Digest: `sha256:288252092889a52a2e3f1c0087e4a380a601beedf54615c18312ab59cf1f3fb5`
- OS: Ubuntu 20.04, `linux/amd64`
- CUDA: 12.1.0 with CUDA development tools
- Conda env: `/opt/conda/envs/my`
- Python: 3.8
- PyTorch stack:
  - `torch==2.1.0+cu121`
  - `torchvision==0.16.0+cu121`
  - `torchaudio==2.1.0+cu121`
- Native/source packages:
  - `pybind11 v2.10.0`
  - `Eigen 3.4.0`
  - `pytorch3d` from `facebookresearch/pytorch3d@stable`
  - `kaolin` from `NVIDIAGameWorks/kaolin`
  - `nvdiffrast` from `NVlabs/nvdiffrast`

Note: `FoundationPose/requirements.txt` currently pins `torch==2.4.1` and
`torchvision==0.19.1`. That is not the Docker image environment. If the goal is
to reproduce the Docker image, use the PyTorch 2.1.0 CUDA 12.1 stack below.

## System Requirements

Recommended host requirements:

- NVIDIA GPU driver compatible with CUDA 12.1, preferably `>=525`
- Linux x86_64
- A compiler toolchain. Ubuntu 20.04 is closest to the Docker image; Ubuntu
  22.04 should also work, but package names can differ.

Install system build and graphics dependencies on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential gcc g++ gfortran cmake ninja-build git wget curl bzip2 \
  pkg-config yasm checkinstall \
  libboost-all-dev libeigen3-dev libgtk2.0-dev libtbb-dev libatlas-base-dev \
  libjpeg-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
  libdc1394-dev libxine2-dev libv4l-dev v4l-utils \
  libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
  libgphoto2-dev libhdf5-dev doxygen libflann-dev \
  proj-data libproj-dev libyaml-cpp-dev cmake-curses-gui \
  libzmq3-dev freeglut3-dev libgl1 libegl1
```

On Ubuntu 22.04, `qt5-default` is usually unavailable. Use `qtbase5-dev` if a
Qt package is needed.

## Create Conda Environment

```bash
conda create -n foundationposepp python=3.8 -y
conda activate foundationposepp

# disabled: use system provide
# conda install -y -c conda-forge \
#   cmake ninja eigen=3.4.0 boost-cpp pybind11 h5py

# disabled: use system provide version of cuda-12.1
# Provides nvcc and CUDA headers inside the conda environment.
# conda install -y -c nvidia cuda-toolkit=12.1

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

It is useful to persist these variables after the environment is verified:

```bash
# conda env config vars set CUDA_HOME=$CONDA_PREFIX
# conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda deactivate
conda activate foundationposepp
```

## Install PyTorch

Use the versions from the Docker image:

```bash
pip install \
  torch==2.1.0+cu121 \
  torchvision==0.16.0+cu121 \
  torchaudio==2.1.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Quick check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

## Install FoundationPose Dependencies

From the repository root:

```bash
export PROJECT_ROOT=/mnt/homes/xinyu-ldap/teleop_platform_ws/teleop_platform_realman/ext/FoundationPose-plus-plus
cd $PROJECT_ROOT/FoundationPose
```

Install the Python dependencies. Keep the PyTorch packages already installed
above:

```bash
pip install -r requirements.txt --no-deps

pip install \
  numpy==1.26.4 scipy==1.12.0 scikit-learn==1.4.1.post1 \
  scikit-image==0.22.0 pyyaml==6.0.1 ruamel.yaml==0.18.6 \
  imageio==2.34.0 opencv-python==4.9.0.80 \
  opencv-contrib-python==4.9.0.80 open3d==0.18.0 \
  trimesh==4.2.2 xatlas==0.0.9 rtree==1.2.0 \
  pyrender==0.1.45 "PyOpenGL>=3.1.0" "PyOpenGL_accelerate>=3.1.0" \
  transformations==2024.6.1 warp-lang==1.0.2 einops==0.7.0 \
  kornia==0.7.2 pandas matplotlib pillow psutil tqdm requests \
  pyzmq msgpack msgpack-numpy fastapi uvicorn hydra-core pydantic
```

Some packages in `requirements.txt` are only needed for training, notebooks, or
auxiliary tools. If a package fails and is not used by your workflow, install it
later on demand.

## Install Native Rendering Packages

These packages are compiled against the current PyTorch and CUDA toolkit.

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install "git+https://github.com/NVlabs/nvdiffrast.git"

git clone --recursive https://github.com/NVIDIAGameWorks/kaolin /tmp/kaolin
cd /tmp/kaolin
FORCE_CUDA=1 pip install -e .
```

If a compile step cannot find CUDA, check:

```bash
which nvcc
echo $CUDA_HOME
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)
PY
```

## Build FoundationPose Extensions

```bash
cd $PROJECT_ROOT/FoundationPose
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/pybind11/share/cmake/pybind11:$CONDA_PREFIX \
  bash build_all_conda.sh
```

Important: the current local checkout does not contain
`FoundationPose/bundlesdf/mycuda`, but `build_all_conda.sh` enters that path.
If this directory is still missing, the build will fail at the mycuda step. Fix
that by restoring the missing BundleSDF `mycuda` directory from the upstream
FoundationPose/BundleSDF source, or by editing the build script if your workflow
does not need that extension.

## Install FoundationPose++ Utilities

### SAM-HQ

```bash
cd $PROJECT_ROOT
pip install segment-anything-hq
pip install -e sam-hq
```

Download the SAM-HQ checkpoint to:

```text
$PROJECT_ROOT/sam-hq/pretrained_checkpoints/sam_hq_vit_h.pth
```

The original install note links to:

```text
https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view
```

### Qwen2-VL

```bash
pip install "git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830" accelerate
pip install qwen-vl-utils
```

Download Qwen2-VL weights to:

```text
$PROJECT_ROOT/Qwen2-VL/weights
```

The original install note uses:

```text
https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
```

### Cutie

```bash
cd $PROJECT_ROOT/Cutie
pip install -e .
python cutie/utils/download_models.py
```

## Download FoundationPose Weights

Download FoundationPose weights to:

```text
$PROJECT_ROOT/FoundationPose/weights
```

The original install note links to:

```text
https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i
```

## Validation

Run this import check first:

```bash
cd $PROJECT_ROOT
python - <<'PY'
import torch
import torchvision
import cv2
import numpy as np
import trimesh
import open3d
import nvdiffrast.torch as dr
import pytorch3d
import kaolin
import zmq
import msgpack_numpy

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("ok")
PY
```

Then run the demo or your own sequence following `README.md`.

## Common Problems

- `torch` version mismatch: use `torch==2.1.0+cu121` to reproduce the Docker
  image. Do not mix it with `torchvision==0.19.1`.
- `nvcc` not found: install `cuda-toolkit=12.1` in conda or use a system CUDA
  12.1 install, then set `CUDA_HOME`.
- `Eigen3` or `pybind11` not found by CMake: make sure `CMAKE_PREFIX_PATH`
  includes `$CONDA_PREFIX`.
- `FoundationPose/bundlesdf/mycuda` missing: restore that directory before
  running `build_all_conda.sh`.
- Headless rendering errors: install OpenGL/EGL system libraries and set up the
  appropriate backend for the target machine.
