from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import smtplib
from email.message import EmailMessage
import os
import random

# Load environment variables
EMAIL = os.getenv("GMAIL_USER")
PASSWORD = os.getenv("GMAIL_PASS")
RECIPIENT = os.getenv("TO_EMAIL")

assert EMAIL and PASSWORD and RECIPIENT, "❌ Missing environment variables!"

# Define a list of dynamic prompts
PROMPTS = [
    "portrait of a beautiful Indian model, white background, 8k, studio lighting",
    "cinematic photo of a fashion model in natural light, high detail, Vogue style",
    "close-up of a model with traditional Indian jewelry, bokeh background, sharp focus",
    "elegant woman in a modern saree, editorial photo, clean white backdrop, 8k",
    "model posing under soft lighting, high-resolution, professional photo shoot"
]

def generate_image(prompt, seed=1234):
    print("⏳ Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32).to("cpu") 

    # Optional offload attempt
    try:
        pipe.enable_model_cpu_offload()
    except ImportError:
        print("⚠️ CPU offload not available or failed.")

    print(f"🎨 Generating image for prompt: {prompt}")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save("output.png")
    print("✅ Image saved as output.png")

def send_email(prompt_used):
    print("📧 Preparing email...")
    msg = EmailMessage()
    msg["Subject"] = "Your Daily AI Model Image"
    msg["From"] = EMAIL
    msg["To"] = RECIPIENT
    msg.set_content(f"Here is your daily AI-generated model image.\n\nPrompt used:\n{prompt_used}")

    with open("output.png", "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="png", filename="model.png")

    print("🚀 Sending email...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL, PASSWORD)
        smtp.send_message(msg)
    print("✅ Email sent to", RECIPIENT)

if __name__ == "__main__":
    selected_prompt = random.choice(PROMPTS)
    generate_image(prompt=selected_prompt)
    send_email(selected_prompt)
    os.remove("output.png")
