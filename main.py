import os
import torch
from diffusers import StableDiffusionPipeline
from instagrapi import Client
import smtplib
from email.message import EmailMessage

# Force torch to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def generate_image(prompt, seed=1234):
    print("⏳ Loading Stable Diffusion model on CPU...")
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to("cpu")

    print(f"🎨 Generating image for prompt: {prompt}")
    generator = torch.Generator().manual_seed(seed)
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save("output.png")
    print("✅ Image saved as output.png")

def post_to_instagram(username, password, caption=""):
    cl = Client()
    cl.login(username, password)
    cl.photo_upload("output.png", caption)
    print("📸 Posted to Instagram!")

def send_email(subject, body, sender_email, sender_password, recipient_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    with open("output.png", "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="png", filename="output.png")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
    print("📧 Email sent successfully!")

if __name__ == "__main__":
    from os import getenv

    prompt = "portrait of a beautiful model, white background, studio light, 8k, ultra detailed"

    generate_image(prompt)

    post_to_instagram(
        username=getenv("IG_USERNAME"),
        password=getenv("IG_PASSWORD"),
        caption=prompt
    )

    send_email(
        subject="🖼️ Your AI Image of the Day",
        body="See the attached image.",
        sender_email=getenv("EMAIL_ADDRESS"),
        sender_password=getenv("EMAIL_PASSWORD"),
        recipient_email=getenv("RECIPIENT_EMAIL")
    )
