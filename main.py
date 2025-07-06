from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import smtplib
from email.message import EmailMessage
import os
from instagrapi import Client

# Load environment variables
EMAIL = os.getenv("GMAIL_USER")
PASSWORD = os.getenv("GMAIL_PASS")
RECIPIENT = os.getenv("TO_EMAIL")
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")

def generate_image(prompt, seed=1234):
    print("⏳ Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.enable_model_cpu_offload()
    pipe = pipe.to("cpu")

    print(f"🎨 Generating image for prompt: {prompt}")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save("output.png")
    print("✅ Image saved as output.png")

def post_to_instagram(caption="Daily AI model image"):
    print("📸 Posting to Instagram...")
    cl = Client()
    cl.login(IG_USERNAME, IG_PASSWORD)
    cl.photo_upload("output.png", caption)
    print("✅ Posted to Instagram")

def send_email():
    print("📧 Preparing email...")
    msg = EmailMessage()
    msg["Subject"] = "Your Daily AI Model Image"
    msg["From"] = EMAIL
    msg["To"] = RECIPIENT
    msg.set_content("Here is your daily AI-generated model image.")

    with open("output.png", "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="png", filename="model.png")

    print("🚀 Sending email...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL, PASSWORD)
        smtp.send_message(msg)
    print("✅ Email sent to", RECIPIENT)

if __name__ == "__main__":
    prompt = "portrait of a beautiful model, white background, studio light, 8k, ultra detailed"
    generate_image(prompt)
    post_to_instagram(caption=prompt)
    send_email()
