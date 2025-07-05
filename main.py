import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os

def generate_image():
    API_URL = "https://api-inference.huggingface.co/models/dreamlike-art/dreamlike-photoreal-2.0"
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}
    payload = {
        "inputs": "portrait photo of a young woman with long wavy brunette hair, soft smile, wearing a red dress, studio lighting, photorealistic",
        "parameters": {
            "seed": 12345,
            "guidance_scale": 7.5,
            "num_inference_steps": 30
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        with open("model.jpg", "wb") as f:
            f.write(response.content)
        print("Image generated successfully.")
    else:
        raise Exception("Failed to generate image: " + response.text)

def send_email():
    msg = MIMEMultipart()
    msg["Subject"] = "Your Daily AI Model 💫"
    msg["From"] = os.environ["GMAIL_USER"]
    msg["To"] = os.environ["TO_EMAIL"]

    with open("model.jpg", "rb") as img:
        img_data = MIMEImage(img.read(), name="model.jpg")
        msg.attach(img_data)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["GMAIL_USER"], os.environ["GMAIL_PASS"])
        server.sendmail(msg["From"], msg["To"], msg.as_string())
        print("Email sent successfully.")

if __name__ == "__main__":
    generate_image()
    send_email()
