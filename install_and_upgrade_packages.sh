#!/bin/bash

# 환경 변수 설정
export BNB_CUDA_VERSION=
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.3/lib64

# 필요한 패키지 목록
packages=(
    "diffusers"
    "transformers"
    "accelerate"
    "safetensors"
    "invisible_watermark"
    "bitsandbytes"
    "opencv-python-headless"
    "peft"
    "huggingface_hub"
)

# 기존 패키지 제거 (필요한 경우)
pip uninstall -y bitsandbytes opencv-python-headless opencv-python

# 패키지 설치 및 업그레이드
pip install --upgrade "${packages[@]}"

# PyTorch 설치 (CUDA 12.3 호환)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu123

# transformer_engine 및 관련 패키지 설치
pip install transformer_engine

# 설치 확인
echo "Installed packages:"
pip list | grep -E 'torch|transformers|accelerate|diffusers|safetensors|paramiko|scp|transformer_engine'

# 환경 변수 설정 확인
echo "Environment variables:"
echo "BNB_CUDA_VERSION: $BNB_CUDA_VERSION"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
