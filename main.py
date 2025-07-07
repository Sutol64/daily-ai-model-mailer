import random
import smtplib
import torch
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from diffusers import StableDiffusionPipeline

# Prompt list
prompts = [
    "portrait of a beautiful Indian model, white background, 8k, studio lighting",
    "cinematic photo of a fashion model in natural light, high detail, Vogue style",
    "studio portrait of a South Asian model, high-resolution, elegant lighting",
]

# Pick a random prompt
selected_prompt = random.choice(prompts)

def generate_image(prompt, seed=1234):
    print("⏳ Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"

    # Load model in CPU mode
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    pipe.to("cpu")

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
    body = "Here is your AI-generated model image for today."

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

    print("✅ Email sent successfully.")

# Main workflow
if __name__ == "__main__":
    generate_image(prompt=selected_prompt)
    send_email()
