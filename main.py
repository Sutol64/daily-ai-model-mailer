import os
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from huggingface_hub import login
import smtplib
from email.message import EmailMessage
from datetime import datetime

# === CONFIG ===
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "AiLotus/woman877-lora/Woman877.v2.safetensors"
OUTPUT_IMAGE = "output.png"
PROMPT = "portrait of a beautiful Indian woman, looking at camera, ultra detailed, natural skin texture, soft lighting"
NEGATIVE_PROMPT = "blurry, low quality, watermark"

# === LOGIN to Hugging Face ===
login(token=os.environ["HF_TOKEN"])

# === GENERATE IMAGE ===
def generate_image(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # fallback to float32 for CPU
    )
    pipe = pipe.to(device)

    # Load LoRA weights
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=MODEL_ID,
        weight_name=LORA_PATH
    )

    # Enable LoRA
    pipe.fuse_lora()

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    image.save(OUTPUT_IMAGE)
    return OUTPUT_IMAGE

# === SEND EMAIL ===
def send_email(image_path):
    user = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_PASS"]
    to_email = os.environ["TO_EMAIL"]

    msg = EmailMessage()
    msg["Subject"] = f"Daily AI Image - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = user
    msg["To"] = to_email
    msg.set_content("Attached is your daily AI-generated image.")

    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        msg.add_attachment(img_data, maintype="image", subtype="png", filename=image_path)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)

# === MAIN ===
def main():
    print("Generating image...")
    image_path = generate_image(PROMPT)
    print("Sending email...")
    send_email(image_path)
    print("Done!")

if __name__ == "__main__":
    main()
