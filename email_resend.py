import resend

resend.api_key = "your-api-key"
counter = 0


async def send_email(email_address, image_data, email_text, reset_counter):
    global counter
    if reset_counter:
        counter = 0
    counter += 1
    params = {
        "from": "Image Bot 🤖 <images@your-connected-domain>",
        "to": email_address,
        "cc": "some_email_address",
        "subject": f"KI Bild",
        "text": email_text,
        "attachments": [
            {"filename": f"image_{counter}.png", "content": list(image_data)}
        ],
    }
    resend.Emails.send(params)
