import os
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_REPO = "AiLotus/woman877-lora"  # Must be SD 1.5 compatible
LORA_FILE = "Woman877.v1_sd15.safetensors"  # Must match the LoRA format
PROMPT = "beautiful Indian woman, ultra-detailed portrait, 4K, soft light"
OUTPUT_DIR = "outputs"

def generate_image(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)
    pipe.fuse_lora()

    image = pipe(prompt=prompt).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(filename)
    return filename

if __name__ == "__main__":
    generate_image(PROMPT)
