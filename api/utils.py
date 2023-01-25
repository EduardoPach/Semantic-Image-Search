from __future__ import annotations

import os
import shutil
import requests
from io import BytesIO
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

def compute_embedding(
    model: CLIPModel, 
    processor: CLIPProcessor, 
    batch: Union[list[Image.Image], list[str]], 
    embedding_type: str="visual",
    device: str="cpu"
) -> np.array:
    """_summary_

    Parameters
    ----------
    model : CLIPModel
        _description_
    processor : CLIPProcessor
        _description_
    batch : Union[list[Image.Image], list[str]]
        _description_
    embedding_type : str, optional
        _description_, by default "visual"
    device : str, optional
        _description_, by default "cpu"

    Returns
    -------
    torch.Tensor
        _description_
    """
    model.to(device)
    if embedding_type=="visual":
        processed_images = processor(images=batch, return_tensors="pt").to(device)
        return model.get_image_features(**processed_images).detach().cpu().numpy()
    elif embedding_type=="text":
        processed_text = processor(text=batch, return_tensors="pt").to(device)
        return model.get_text_features(**processed_text).detach().cpu().numpy()

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

def search_image_from_text(
    model: CLIPModel, 
    processor: CLIPProcessor, 
    query: str,
    k: int=15,
    embedding_dir: str="./embeddings"
) -> list[Image.Image]:
    """Get's the top k images most similar to the 
    text query.

    Parameters
    ----------
    model : CLIPModel
        A CLIP model
    processor : CLIPProcessor
        Processor for CLIP
    query : str
        A text query to be matched with the images
    k : int, optional
        The number of images to be retrieved, by default 15
    embedding_dir : str, optional
        Path to directory where embeddings are stored, by default "./embeddings"

    Returns
    -------
    list[Image.Image]
        List of the top k images most similar with the text query
    """
    embedding_dir = Path(embedding_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Images embeddings and URLs
    embedding_list = [np.load(embedding_file) for embedding_file in sorted(embedding_dir.glob("*.npy"))]
    url_list = [pd.read_csv(url_file) for url_file in sorted(embedding_dir.glob("*.csv"))]
    urls_df = pd.concat(url_list).reset_index(drop=True)
    embeddings = np.concatenate(embedding_list)
    # Query embedding
    text_emb = compute_embedding(model, processor, [query], "text", device)
    text_emb /= np.linalg.norm(text_emb)
    # Getting Similarities
    similarities = cosine_similarity(embeddings, text_emb.T.reshape(1, -1))
    top_k_indices = similarities.flatten().argsort()[-k:]
    # Getting list of Images
    urls = urls_df.loc[top_k_indices, "url"].tolist()

    return [fetch_image(url)[-1] for url in urls]