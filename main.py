import os
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import torch
from datetime import datetime

PROMPT = "A beautiful Indian woman in traditional attire, 4K, studio lighting"
OUTPUT_DIR = "outputs"
LORA_REPO = "AiLotus/woman877-lora"
LORA_FILENAME = "Woman877.v2.safetensors"

def generate_image(prompt: str) -> str:
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load LoRA
    print("Loading LoRA weights...")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILENAME)

    # Use LoRA
    pipe.fuse_lora()

    # Generate image
    print("Generating image...")
    image = pipe(prompt=prompt).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(filename)

    print(f"Image saved to: {filename}")
    return filename

def main():
    print("Starting image generation pipeline...")
    path = generate_image(PROMPT)
    print("Done.")

if __name__ == "__main__":
    main()
