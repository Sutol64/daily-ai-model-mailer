import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
import random
from datetime import datetime

# Authenticate with Hugging Face token from secrets (replace in GitHub Actions)
HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]
login(HUGGINGFACE_TOKEN)

# === CONFIG ===
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_REPO = "AiLotus/woman877-lora"
LORA_FILE = "Woman877.v2.safetensors"
PROMPTS = [
    "beautiful Indian woman, traditional attire, DSLR photo, portrait, high quality",
    "fashion model in sari, cinematic lighting, soft focus, Vogue style",
    "young woman, golden hour, outdoors, natural smile, realistic",
]

def generate_image(prompt):
    print(f"Generating image with prompt: {prompt}")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cpu")

    # Load LoRA weights
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)

    # Generate image
    image = pipe(prompt).images[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image.save(f"output-{timestamp}.png")
    print("✅ Image saved successfully.")

if __name__ == "__main__":
    selected_prompt = random.choice(PROMPTS)
    generate_image(selected_prompt)
