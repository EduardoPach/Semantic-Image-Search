from __future__ import annotations

import os
import shutil
import requests
from io import BytesIO
from pathlib import Path
from typing import Union

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


class ImageBatchGenerator:
    """  A generator class that takes a list of URLs (associated with images) and a batch size 
    as arguments and yields batches of images when iterated over. 

    Parameters
    ----------
    urls : list[str]
        List of URLs
    batch_size : int, optional
        Number of URLs to yield in each batch, by default 32
    """
    def __init__(self, urls: list[str], batch_size: int=32) -> None:
        self.urls = urls
        self.batch_size = batch_size
        self.current_index = 0

    @staticmethod
    def _load_image(self, url: str) -> Image.Image:
        img_bytes = requests.get(url).content
        return Image.open(BytesIO(img_bytes))

    def __len__(self) -> int:
        return self.batch_size

    def __iter__(self) -> ImageBatchGenerator:
        return self
    
    def __next__(self) -> list[Image.Image]:
        if self.current_index >= len(self.urls):
            raise StopIteration()
        
        start = self.current_index
        end = min(start + self.batch_size, len(self.urls))
        self.current_index = end
        return [self._load_image(url) for url in self.urls[start:end]]

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
    for batch_idx, batch in enumerate(loop):
        embedding = compute_embedding(model, processor, batch, device=device).detach().numpy()
        embedding_name = embeddings_temp_dir / f"{batch_idx:05d}.npy"
        if not embeddings_temp_dir.exists():
            os.mkdir(embeddings_temp_dir)
        np.save(embedding_name, embedding)
    embedding_list = [np.load(embedding_file) for embedding_file in sorted(embeddings_temp_dir.glob("*.npy"))]
    embeddings = np.concatenate(embedding_list)
    if not embeddings_dir.exists():
        os.mkdir(embeddings_dir)
    np.save(embeddings_dir / (embedding_file_name+'.npy'), embeddings)
    shutil.rmtree(embeddings_temp_dir)
    