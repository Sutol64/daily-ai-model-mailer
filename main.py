import os
from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime

PROMPT = "A beautiful Indian woman in traditional attire, 4K, studio lighting"
OUTPUT_DIR = "outputs"
LORA_REPO = "AiLotus/woman877-lora"
LORA_FILENAME = "Woman877.v2.safetensors"

def generate_image(prompt: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device} with dtype: {dtype}")

    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    pipe.to(device)

    # Load LoRA
    print("Loading LoRA weights...")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILENAME)
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
    generate_image(PROMPT)
    print("Done.")

if __name__ == "__main__":
    main()
