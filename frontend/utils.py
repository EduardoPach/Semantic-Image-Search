from __future__ import annotations

import base64
import requests

import streamlit as st

def fetch_image_bytes(url: str) -> bytes:
    """This function fetches image bytes from url.

    Parameters
    ----------
    url : str
        URL of the image

    Returns
    -------
    bytes
        Fetched image bytes
    """
    return requests.get(url).content

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

def start_sidebar() -> tuple[str, int]:
    with st.sidebar.container():
        c1, c2 = st.columns(2)
        mode = c1.radio("Query Mode", ["Text", "Image"])
        k = c2.slider("Number of Images", min_value=1, max_value=20, step=1, value=10)
        if mode == "Text":
            query = st.text_input("Query", value="Two dogs playing")
        else:
            query = None
            uploaded_image = st.file_uploader("Query", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                query = encode_image(uploaded_image.read()) 
                st.image(uploaded_image, use_column_width=True, caption="Query Image")
        
        return query, k

def refined_search(url: str, k) -> list[str]:
    """Refined search using the clicked image.

    Parameters
    ----------
    url : str
        URL of the clicked image
    k : _type_
        Number of images to return

    Returns
    -------
    list[str]
        List of image URLs most similar to the clicked image
    """
    query = fetch_image_bytes(url)
    query = encode_image(query)
    img_urls = send_request(query, k)
    return img_urls

