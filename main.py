import os
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from email.message import EmailMessage
from datetime import datetime
import smtplib

# === CONFIG ===
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_REPO = "AiLotus/woman877-lora"
LORA_FILENAME = "Woman877.v2.safetensors"
PROMPT = "portrait of a beautiful Indian woman, ultra realistic, soft lighting, looking at camera"
NEGATIVE_PROMPT = "blurry, low quality, watermark"
OUTPUT_IMAGE = "output.png"

# === AUTH ===
login(token=os.environ["HF_TOKEN"])

# === GENERATE IMAGE ===
def generate_image(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    # Load LoRA
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILENAME)
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
    msg = EmailMessage()
    msg["Subject"] = f"Daily AI Image - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = os.environ["GMAIL_USER"]
    msg["To"] = os.environ["TO_EMAIL"]
    msg.set_content("Attached is your daily AI-generated image.")

    with open(image_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="png", filename=image_path)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.environ["GMAIL_USER"], os.environ["GMAIL_PASS"])
        smtp.send_message(msg)

# === MAIN ===
def main():
    print("Generating image...")
    path = generate_image(PROMPT)
    print("Sending email...")
    send_email(path)
    print("Done!")

if __name__ == "__main__":
    main()
