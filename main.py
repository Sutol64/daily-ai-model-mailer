from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from instagrapi import Client
import os

# Load environment variables from GitHub secrets
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")

def generate_image(
    prompt="portrait of a beautiful model, white background, studio light, 8k, ultra detailed",
    seed=1234
):
    print("⏳ Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, cache_dir="./model_cache")
    pipe.enable_model_cpu_offload()
    pipe = pipe.to("cpu")

    print(f"🎨 Generating image for prompt: {prompt}")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save("output.png")
    print("✅ Image saved as output.png")

def post_to_instagram():
    print("📸 Logging into Instagram...")
    cl = Client()
    cl.login(IG_USERNAME, IG_PASSWORD)

    print("🚀 Posting to Instagram...")
    cl.photo_upload(
        path="output.png",
        caption="Your daily AI-generated model photo. #AI #StableDiffusion"
    )
    print("✅ Posted successfully!")

if __name__ == "__main__":
    generate_image()
    post_to_instagram()
