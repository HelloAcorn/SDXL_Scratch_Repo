#!/bin/bash

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
