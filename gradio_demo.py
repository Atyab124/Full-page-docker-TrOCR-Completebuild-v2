import gradio as gr
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

def api_interaction(image_pil):
    # Convert PIL Image to bytes
    image_byte_array = BytesIO()
    image_pil.save(image_byte_array, format="JPEG")  # Adjust format as needed

    url = "http://172.17.0.2:8080/hand_OCR"  # Replace with your API URL
    files = {'image': ('image.jpg', image_byte_array.getvalue())}
    response = requests.post(url, files=files)
    return response.text# .json()["result"] 

iface = gr.Interface(
    fn=api_interaction,
    inputs=gr.inputs.Image(type="pil"),
    outputs="text",
    live=True,
    layout="vertical",
    capture_session=True
)

load_dotenv()

if int(os.environ['SHARE'])==1:
    iface.queue().launch(share=True)

elif int(os.environ['SHARE'])==0:
    iface.queue().launch(share=False)

else:
    print("Invalid .env variable 'SHARE', please set value '1' for True or '0' for False")