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
    model.to(device)
    if embedding_type=="visual":
        processed_images = processor(images=batch, return_tensors="pt").to(device)
        return model.get_image_features(**processed_images)
    elif embedding_type=="text":
        processed_text = processor(text=batch, return_tensors="pt").to(device)
        return model.get_text_features(**processed_text)

