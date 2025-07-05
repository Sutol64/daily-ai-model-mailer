import os
import replicate
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime

# === CONFIG ===
receiver = os.getenv("TO_EMAIL")  # Replace with actual recipient

# === IMAGE GENERATION ===
def generate_image():
    print("Generating image from Replicate...")

    replicate_api = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_api:
        raise Exception("REPLICATE_API_TOKEN not found in environment variables")

    replicate_client = replicate.Client(api_token=replicate_api)

    output = replicate_client.run(
        "lucataco/realistic-vision-v5.1:29fc1740a7396f8f3cd90cdbdd4659eea8b3b380ef63fcfcd905f7f4c7d67d76",
        input={
            "prompt": "portrait of a beautiful confident woman, symmetrical face, detailed eyes, medium-long hair, natural skin tone, high detail, fashion look, realistic lighting",
            "negative_prompt": "extra limbs, distorted hands, low quality, blur, cartoon",
            "width": 512,
            "height": 768,
            "guidance_scale": 7,
            "num_inference_steps": 30,
            "seed": 12345  # Keeps image consistent
        }
    )

    if not output or not output[0].startswith("http"):
        raise Exception("Failed to generate image")

    image_url = output[0]
    print(f"Image generated: {image_url}")

    # Download image
    import requests
    response = requests.get(image_url)
    image_path = "model_image.png"
    with open(image_path, "wb") as f:
        f.write(response.content)

    return image_path

# === EMAIL ===
def send_email(image_path):
    print("Sending email with image...")

    sender = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender or not password:
        raise Exception("Missing Gmail credentials in environment")

    msg = EmailMessage()
    msg["Subject"] = f"Your Daily AI Model - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content("Attached is today's AI-generated model image.")

    with open(image_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="png", filename="model.png")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)

    print("Email sent successfully.")

# === MAIN ===
if __name__ == "__main__":
    try:
        image_path = generate_image()
        send_email(image_path)
    except Exception as e:
        print(f"Error: {e}")
