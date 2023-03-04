from __future__ import annotations

import requests
from io import BytesIO
from typing import Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def load_model(model_id: str) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def fetch_image(url: str) -> tuple[str, Image.Image]:
    """Fetch image from url

    Parameters
    ----------
    url : str
        url of the image

    Returns
    -------
    tuple[str, Image.Image]
        tuple (url, image) where image is PIL image object and url is the url of the image
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return url, Image.open(BytesIO(response.content))
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Something Else: {err}")
    return url, None

def index_to_url(index: list[int], url_table: pd.DataFrame) -> list[str]:
    """Converts a list of indices to a list of urls

    Parameters
    ----------
    index : list[int]
        List of indices
    url_table : pd.DataFrame
        DataFrame containing the urls

    Returns
    -------
    list[str]
        List of urls
    """
    return url_table.iloc[index].url.tolist()

class ImageBatchGenerator:
    """
    A generator class that get's as arguments a list of URLs and batch size and generates batches of PIL images
    that are obtained through GET requests to the URLs.

    Parameters
    ----------
    urls : list[str]
        List of URLs to fetch images from
    batch_size : int
        The size of the batches to be generated
    """
    def __init__(self, urls: list[str], batch_size: int=32) -> None:
        self.urls = urls
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor()
        self.futures = self.executor.map(fetch_image, self.urls)
    
    def __len__(self) -> int:
        return (len(self.urls) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> ImageBatchGenerator:
        return self

    def __next__(self) -> dict[str, Union[str, Image.Image]]:
        images = []
        urls = []
        for future in self.futures:
            url, image = future
            if image is not None:
                images.append(image)
                urls.append(url)
            if len(images) == self.batch_size:
                break
        if len(images) == 0:
            self.executor.shutdown()
            raise StopIteration
        return {"images": images, "urls": urls}
    
