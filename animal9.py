import os
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.utils import make_image_grid
from PIL import Image

# 경로 설정
base_ckpt_path = "/home/work/safetensors/cartoonxl_v10.safetensors"

# 기본 모델 로드
pipe = StableDiffusionXLPipeline.from_single_file(base_ckpt_path, torch_dtype=torch.float16).to('cuda')

# 리파이너 모델 로드
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

# 동물 리스트
animals = ["rabbit", "cat", "dog", "panda", "raccoon", "hedgehog", "squirrel", "duck", "hamster"]
prompts_template = "{}, crying a starry night, sad, content, forest background"
negative_prompt = "multiple tails, more than one tail, two tails, extra tails, three ears, four ears, extra ears, more than two ears, distorted body, misshapen body, disproportionate body, irregular body, twisted body, warped body, unnatural body"
steps = 35

images = []

# 각 동물에 대해 이미지 생성 및 리파인
for animal in animals:
    prompts = prompts_template.format(animal)
    
    base_image = pipe(
        prompt=prompts, 
        negative_prompt=negative_prompt, 
        num_inference_steps=steps, 
        guidance_scale=7.2, 
    )
    
    latent_image = base_image.images[0]

    refine_image = refiner(
        prompt=prompts,
        num_inference_steps=steps,
        denoising_start=0.8,
        image=latent_image
    )
    
    images.append(refine_image.images[0])

# 이미지 그리드로 합치기
grid_image = make_image_grid(images, rows=3, cols=3)
grid_image.show()

# 이미지 저장
grid_image.save("cute_animals_grid.png")
