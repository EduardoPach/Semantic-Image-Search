from __future__ import annotations

import base64
import requests

import streamlit as st

def send_request(query: str, k: int) -> list[str]:
    """Send request to backend and return list of image URLs."""
    try:
        url = "http://localhost:8000/search"
        payload = {"query": [query], "k": k}
        response = requests.post(url, json=payload)
    except:
        url = "http://backend-api:8000/search"
        payload = {"query": [query], "k": k}
        response = requests.post(url, json=payload)
    return response.json()["urls"][0]

def encode_image(image_bytes: bytes) -> str:
    """Encode image to base64."""
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"