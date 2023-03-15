from __future__ import annotations

import requests

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