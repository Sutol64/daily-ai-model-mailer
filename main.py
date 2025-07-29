# main.py
import os
import torch
from diffusers import StableDiffusionXLPipeline
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

def generate_image(prompt):
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=HF_TOKEN,
        variant="fp16"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    pipe.load_lora_weights(
        pretrained_model_name_or_path="AiLotus/woman877-lora",
        weight_name="Woman877.v2.safetensors",
        token=HF_TOKEN,
    )

    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)

    image = pipe(prompt=prompt).images[0]
    filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(filename)
    return filename

def send_email(image_path):
    from_email = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_PASS")
    to_email = os.getenv("TO_EMAIL")

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = "Your Daily AI-Generated Image"

    with open(image_path, 'rb') as f:
        mime = MIMEBase('image', 'png', filename=image_path)
        mime.add_header('Content-Disposition', 'attachment', filename=image_path)
        mime.set_payload(f.read())
        encoders.encode_base64(mime)
        msg.attach(mime)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.send_message(msg)
    server.quit()

def main():
    prompt = "portrait of a beautiful woman, 1024x1024, realistic lighting"
    image_path = generate_image(prompt)
    send_email(image_path)

if __name__ == "__main__":
    main()
