from __future__ import annotations

from typing import Union

import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image

from api.utils import *

class SemanticSearcher:
    """Object that performs semantic search on images and text

    Parameters
    ----------
    model_id : str
        HuggingFace model id for MultiModal model
    index : faiss.Index, optional
        Faiss index with embeddings to search, by default None
    index_to_path : pd.DataFrame, optional
        DataFrame containing the image paths, by default None
    """
    def __init__(self, model_id: str, index: faiss.Index=None, index_to_path: pd.DataFrame=None) -> None:
        self.model, self.processor = load_model(model_id)
        self.index = index
        self.index_to_path = index_to_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(self, batch: Union[list[Image.Image], list[str]]) -> np.array:
        """Process a batch of images or text to extract their embeddings

        Parameters
        ----------
        batch : Union[list[Image.Image], list[str]]
            Batch containing images or text

        Returns
        -------
        np.array
            Resulting batch embeddings
        """
        self.model.to(self.device)
        mode = self._infer_type(batch)
        if mode=="visual":
            processed_images = self.processor(images=batch, return_tensors="pt").to(self.device)
            return self.model.get_image_features(**processed_images).detach().cpu().numpy()
        elif mode=="text":
            processed_text = self.processor(text=batch, return_tensors="pt").to(self.device)
            return self.model.get_text_features(**processed_text).detach().cpu().numpy()

    def __call__(self, query: Union[list[Image.Image], list[str]], k: int=5) -> list:
        """Perform a semantic search on a batch of images or text

        Parameters
        ----------
        query : Union[list[Image.Image], list[str]]
            THe input query used to perform the search
        k : int, optional
            Number of items return from query, by default 5

        Returns
        -------
        np.array
            An array containing the indexes of the k most similar items
        """
        # Query embedding
        query_emb = self.process([query])
        query_emb /= np.linalg.norm(query_emb)
        # Getting Similarities
        _, I = self.index.search(query_emb, k)
        
        return index_to_path(I.flatten().tolist(), self.index_to_path.copy())

    @staticmethod
    def _infer_type(x: Union[list[Image.Image], list[str]]) -> str:
        """Infers the type of the input batch

        Parameters
        ----------
        x : Union[list[Image.Image], list[str]]
            Input batch

        Returns
        -------
        str
            Type of the input batch
        """
        return "visual" if isinstance(x[0], Image.Image) else "text"