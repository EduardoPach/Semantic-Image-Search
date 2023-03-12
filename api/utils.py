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

def fetch_image(img_path: str) -> tuple[str, Image.Image]:
    """Fetch image from img_path

    Parameters
    ----------
    img_path : str
        img_path of the image

    Returns
    -------
    tuple[str, Image.Image]
        tuple (img_path, image) where image is PIL image object and img_path is the img_path of the image
    """
    return img_path, Image.open(img_path)

def index_to_path(index: list[int], img_path_table: pd.DataFrame) -> list[str]:
    """Converts a list of indices to a list of urls

    Parameters
    ----------
    index : list[int]
        List of indices
    img_path_table : pd.DataFrame
        DataFrame containing the urls

    Returns
    -------
    list[str]
        List of image paths
    """
    return img_path_table.iloc[index]['path'].tolist()

class ImageBatchGenerator:
    """
    A generator class that get's as arguments a list of URLs and batch size and generates batches of PIL images
    that are obtained through GET requests to the URLs.

    Parameters
    ----------
    img_paths : list[str]
        List of image paths to fetch images from
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
        img_paths = []
        for future in self.futures:
            img_path, image = future
            if image is not None:
                images.append(image)
                img_paths.append(img_path)
            if len(images) == self.batch_size:
                break
        if len(images) == 0:
            self.executor.shutdown()
            raise StopIteration
        return {"images": images, "urls": img_paths}
    
