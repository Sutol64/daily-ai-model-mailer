import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os

def generate_image():
    API_URL = "https://api-inference.huggingface.co/models/stablediffusionapi/realistic-vision-v51"
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}
    
    prompt = "portrait of a beautiful woman with long wavy brown hair, soft smile, wearing a red dress, studio lighting, photorealistic"
    payload = {
        "inputs": prompt,
        "parameters": {
            "seed": 12345,  # Same seed for consistency
            "guidance_scale": 7.5,
            "num_inference_steps": 30
        }
    }

    print("Generating image from Hugging Face...")
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        with open("model.jpg", "wb") as f:
            f.write(response.content)
        print("✅ Image saved as model.jpg")
    else:
        print("❌ Error generating image:", response.status_code, response.text)
        raise Exception("Image generation failed")

def send_email():
    msg = MIMEMultipart()
    msg["Subject"] = "Your Daily AI Model 😊"
    msg["From"] = os.environ["GMAIL_USER"]
    msg["To"] = os.environ["TO_EMAIL"]

    with open("model.jpg", "rb") as img:
        img_data = MIMEImage(img.read(), name="model.jpg")
        msg.attach(img_data)

    print("Sending email...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["GMAIL_USER"], os.environ["GMAIL_PASS"])
        server.sendmail(msg["From"], msg["To"], msg.as_string())
        print("✅ Email sent successfully!")

if __name__ == "__main__":
    generate_image()
    send_email()
