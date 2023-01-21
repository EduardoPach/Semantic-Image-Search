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

def compute_embedding(
    model: CLIPModel, 
    processor: CLIPProcessor, 
    batch: Union[list[Image.Image], list[str]], 
    embedding_type: str="visual",
    device: str="cpu"
) -> torch.Tensor:
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
        return model.get_image_features(**processed_images)
    elif embedding_type=="text":
        processed_text = processor(text=batch, return_tensors="pt").to(device)
        return model.get_text_features(**processed_text)

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

def compute_dataset_visual_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_generator: ImageBatchGenerator,
    embedding_file_name: str="embeddings",
    embeddings_dir_path: str="./embeddings",
    embeddings_temp_dir_path: str="./temp_embeddings",
    device: str="cpu"
) -> None:
    """
    Compute the visual embeddings for a dataset using a pre-trained CLIP model.

    Parameters
    ----------
    model : CLIPModel
        A pre-trained instance of a CLIP model
    processor : CLIPProcessor
        A instance of the processor that pre-process the images before inference
    batch_generator : ImageBatchGenerator
        An generator object which yield batches of images 
    embedding_file_name : str, optional
        The name of the file to save the embeddings to, by default "embeddings"
    embeddings_dir_path : str, optional
        The directory path where the embeddings will be saved, by default "./embeddings"
    embeddings_temp_dir_path : str, optional
        The directory path where the embeddings will be temporarily saved, by default "./temp_embeddings"
    device : str, optional
        The device to run the inference on. It can be 'cpu' or 'cuda:x' where x is the index of the GPU, by default "cpu"
    """
    loop = tqdm(batch_generator)
    embeddings_temp_dir = Path(embeddings_temp_dir_path)
    embeddings_dir = Path(embeddings_dir_path)
    embedding_file_path = embeddings_dir / (embedding_file_name+'.npy')
    url_file_path = embeddings_dir / (embedding_file_name+'.csv')
    for batch_idx, batch in enumerate(loop):
        images, urls = batch.values()
        embedding = compute_embedding(model, processor, images, device=device).detach().cpu().numpy()
        embedding_name = embeddings_temp_dir / f"{batch_idx:05d}.npy"
        url_names = embeddings_temp_dir / f"{batch_idx:05d}.csv"
        if not embeddings_temp_dir.exists():
            os.mkdir(embeddings_temp_dir)
        np.save(embedding_name, embedding)
        pd.DataFrame(urls, columns=["url"]).to_csv(url_names, index=False)
    embedding_list = [np.load(embedding_file) for embedding_file in sorted(embeddings_temp_dir.glob("*.npy"))]
    url_list = [pd.read_csv(url_file) for url_file in sorted(embeddings_temp_dir.glob("*.csv"))]
    embeddings = np.concatenate(embedding_list)
    urls = pd.concat(url_list).reset_index(drop=True)
    if not embeddings_dir.exists():
        os.mkdir(embeddings_dir)
    if embedding_file_path.exists() and url_file_path.exists():
        curr_embeddings = np.load(embedding_file_path)
        embeddings = np.concatenate([curr_embeddings, embeddings])
        np.save(embeddings_dir / (embedding_file_name+'.npy'), embeddings)
        curr_urls = pd.read_csv(url_file_path)
        urls = pd.concat([curr_urls, urls]).reset_index(drop=True)
        urls.to_csv(url_file_path, index=False)
    else:
        np.save(embedding_file_path, embeddings)
        urls.to_csv(url_file_path, index=False)
    shutil.rmtree(embeddings_temp_dir)
    