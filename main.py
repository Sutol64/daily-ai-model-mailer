from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import smtplib
from email.message import EmailMessage
import os

# Load environment variables from GitHub secrets
EMAIL = os.getenv("GMAIL_USER")
PASSWORD = os.getenv("GMAIL_PASS")
RECIPIENT = os.getenv("TO_EMAIL")

def generate_image(
    prompt="portrait of a beautiful model, white background, studio light, 8k, ultra detailed",
    seed=1234
):
    print("⏳ Loading model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")

    print("🎨 Generating image...")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save("output.png")
    print("✅ Image saved as output.png")

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
    generate_image()
    send_email()
