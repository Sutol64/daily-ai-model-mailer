import os
import random
import smtplib
import torch
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login

# Login to Hugging Face using token from environment
login(token=os.environ["HF_TOKEN"])

# Prompt list
prompts = [
    "portrait of a beautiful woman, trending on instagram, studio light, detailed skin, masterpiece, wearing traditional Indian jewelry, Woman877",
    "Indian model in a cinematic close-up, soft background, Woman877 character style, shallow depth of field, ultra detailed",
    "realistic headshot of Woman877, elegant expression, glowing skin, ultra high resolution, professional lighting"
]
selected_prompt = random.choice(prompts)

def generate_image(prompt, seed=42):
    print("⏳ Loading base Stable Diffusion model...")
    base_model_id = "runwayml/stable-diffusion-v1-5"
    repo_id = "AiLotus/woman877-lora"
    lora_filename = "Woman877.v2.safetensors"

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    )
    pipe.to("cpu")

    # Load and apply LoRA weights
    pipe.load_lora_weights(repo_id, weight_name=lora_filename)
    pipe.fuse_lora()

    print(f"🎨 Generating image for prompt: {prompt}")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save("output.png")
    print("✅ Image saved as output.png")

def send_email():
    print("📧 Sending email...")
    from_email = os.environ["GMAIL_USER"]
    to_email = os.environ["TO_EMAIL"]
    password = os.environ["GMAIL_PASS"]

    subject = "🖼️ AI Generated Model Image"
    body = "Here is your AI-generated Woman877 model image for today."

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with open("output.png", "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=output.png")
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, password)
        server.send_message(msg)

    print("✅ Email sent successfully.")

if __name__ == "__main__":
    generate_image(prompt=selected_prompt)
    send_email()
