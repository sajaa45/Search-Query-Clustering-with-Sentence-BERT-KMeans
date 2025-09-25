import os
import requests
import time

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN =f"Bearer {os.environ['HF_TOKEN']}"
API_URL = "https://api-inference.huggingface.co/models/"
#spanish to english
MODEL_ES_EN = "Helsinki-NLP/opus-mt-es-en"  
#japanese to english
MODEL_JP_EN = "Helsinki-NLP/opus-mt-ja-en"  

headers = {"Authorization": HF_TOKEN}

def query_api(model, text):
    response = requests.post(
        API_URL + model,
        headers=headers,
        json={"inputs": text},
        timeout=30  # same as your test
    )
    if response.status_code != 200:
        # Log the actual error
        print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()

    data = response.json()
    if isinstance(data, list) and "translation_text" in data[0]:
        return data[0]["translation_text"]
    elif "error" in data:
        raise RuntimeError(f"Hugging Face error: {data['error']}")
    else:
        raise RuntimeError(f"Unexpected response format: {data}")


def translate(original_text,source_locale):
    if source_locale=="es":
        return query_api(MODEL_ES_EN, original_text)
    elif source_locale=="jp":
        return query_api(MODEL_JP_EN, original_text)
    else:
        return original_text