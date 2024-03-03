from fastapi import FastAPI, Body, Query
import requests
import replicate
from dotenv import load_dotenv
import os
from google.cloud import storage
import httpx
from datetime import datetime
from aiohttp import ClientSession
from PIL import Image
import io
from random import randint
from google.oauth2.service_account import Credentials
import email_resend

keyfile = {}  # your google keyfile

app = FastAPI()
credentials = Credentials.from_service_account_info(keyfile)
storage_client = storage.Client(
    project="your-google-project-id", credentials=credentials
)
load_dotenv()  # Load .env file
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate.Client(api_token=REPLICATE_API_TOKEN)


async def download_image(image_url):
    async with ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                return await response.read()
            else:
                print(f"Failed to download the image. Status code: {response.status}")


async def upload_image(image_data, bucket_name):
    # Generate a timestamp
    timestamp = datetime.utcnow().strftime("%d-%m-%Y_%H-%M-%S%f")
    random_num = randint(1, 100)
    blob_name = f"{timestamp}-{random_num}.png"
    bucket = storage_client.get_bucket(bucket_name)
    if image_data is not None:
        await upload_to_gcs(bucket, image_data, blob_name)
    else:
        print("Skipping upload due to empty or failed image data.")


async def upload_to_gcs(bucket, data, blob_name):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="image/png")
    print("upload startet")


@app.post("/plugger/request")
async def plugger_request(payload: dict = Body(...)):
    headers = {"x-api-key": "api-key-here"}
    bucket_name = payload.get("bucket")
    response = requests.post(
        "https://inference.plugger.ai/", headers=headers, json=payload
    )
    result = response.json()
    image_data = await download_image(result["data"]["image_url"])
    await upload_image(image_data, bucket_name)
    return {"image_url": result["data"]["image_url"]}


@app.post("/replicate/models")
async def replicate_model_request(payload: dict = Body(...)):
    bucket_name = payload.get("bucket")
    input_prompt = payload.get("input", {}).get("prompt")
    negative_prompt = payload.get("input", {}).get("negative_prompt")
    width = payload.get("input", {}).get("width")
    height = payload.get("input", {}).get("height")
    email = payload.get("email", "")
    reset_counter = payload.get("reset_counter", False)

    model = replicate.models.get(payload.get("model"))
    version = model.versions.get(payload.get("version"))

    # TODO: Upload this FastAPI to Deta and replace the ngrok URL with the Deta URL
    # https://your-deta-app-id.deta.app/replicate/handle_prediction_upload

    if email:
        webhook_url = f"your-ngrok-forward-url/replicate/handle_prediction_email?email={email}&reset_counter={reset_counter}"
    else:
        webhook_url = f"your-ngrok-forward-url/replicate/handle_prediction_upload?bucket={bucket_name}"

    replicate.predictions.create(
        version=version,
        input={
            "prompt": input_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_outputs": 1,
        },
        webhook=webhook_url,
        webhook_events_filter=["completed"],
    )

    return {"response": "Model started"}


@app.post("/replicate/handle_prediction_upload")
async def handle_prediction_upload(data: dict, bucket: str = Query(None)):
    output = data.get("output", None)

    if isinstance(output, list):
        image_url = output[0] if output else None
    else:
        image_url = output

    if image_url:
        # Download the image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True, timeout=10000)
            image_data = response.content

            # Upload the image (assuming upload_image is an async function)
            await upload_image(image_data, bucket)

        return {"image_url": image_url}
    else:
        return {"error": "Output URL is None"}


@app.post("/replicate/handle_prediction_email")
async def handle_prediction_email(
    data: dict, email: str = Query(None), reset_counter: bool = Query(False)
):
    input_data = data.get("input", None)
    output = data.get("output", None)

    if isinstance(output, list):
        image_url = output[0] if output else None
    else:
        image_url = output

    if "image_path" in input_data:
        email_text = f"Image Path: {input_data['image_path']}\Pixel: {input_data['pixel']}\Product Size: {input_data['product_size']}\n\nPrompt: {input_data['prompt']}\n\nNegative Prompt: {input_data['negative_prompt']}"
    else:
        email_text = f"Width: {input_data['width']}\nHeight: {input_data['height']}\nVariants: {input_data['num_outputs']}\n\nPrompt: {input_data['prompt']}\n\nNegative Prompt: {input_data['negative_prompt']}"

    if image_url and email:
        # Download the image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True, timeout=10000)
            image_data = response.content

            # Send email (assuming send_email is an async function)
            await email_resend.send_email(email, image_data, email_text, reset_counter)
            print("email sent")

        return {"image_url": image_url}
    else:
        return {"error": "Output URL is None or email is missing"}


@app.post("/convert")
async def convert_image(payload: dict = Body(...)):
    # Download the image from the URL
    url = payload.get("url")
    bucket_name = payload.get("bucket")

    image_data = await download_image(url)
    image_stream = io.BytesIO(image_data)
    image = Image.open(image_stream)

    # Convert to JPEG
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save to buffer
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    # Get only the second part of the bucket name
    parts = bucket_name.split("-")
    bucket_theme = parts[1] if len(parts) > 1 else None

    # Upload to Google Cloud Storage
    bucket = storage_client.get_bucket(bucket_name)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    blob_name = f"{bucket_theme}-{timestamp}.jpg"
    blob = bucket.blob(blob_name)
    blob.upload_from_file(output_buffer, content_type="image/jpeg")

    return {"message": "Image converted and uploaded to Google Cloud Storage."}
