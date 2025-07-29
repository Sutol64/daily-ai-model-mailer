import os
import random
import smtplib
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from peft import PeftModel, PeftConfig
from transformers import CLIPTextModel, CLIPTokenizer
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

prompts = [
    "photo of woman877, portrait, high detail, soft lighting, studio background, 8k",
    "woman877 standing, natural lighting, photorealistic, high quality",
]

selected_prompt = random.choice(prompts)

def generate_image(prompt, seed=42):
    model_id = "runwayml/stable-diffusion-v1-5"
    repo_id = "AiLotus/woman877-lora"
    lora_filename = "Woman877.v2.safetensors"

    print("⏳ Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None,
        token=os.environ["HF_TOKEN"]
    )
    pipe.to("cpu")

    print("🔗 Loading LoRA weights...")
    pipe.load_lora_weights(repo_id, weight_name=lora_filename, token=os.environ["HF_TOKEN"])
    pipe.fuse_lora()

    print(f"🎨 Generating image for: {prompt}")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, generator=generator, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("output.png")
    print("✅ Saved image as output.png")

def send_email():
    print("📧 Sending email...")
    from_email = os.environ["GMAIL_USER"]
    to_email = os.environ["TO_EMAIL"]
    password = os.environ["GMAIL_PASS"]

    subject = "🖼️ Your AI Model Image"
    body = "Here is your daily AI-generated model image."

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    filename = "output.png"
    with open(filename, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, password)
        server.send_message(msg)
    print("✅ Email sent!")

if __name__ == "__main__":
    generate_image(selected_prompt)
    send_email()
