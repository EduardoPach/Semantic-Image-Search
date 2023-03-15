from __future__ import annotations

from typing import Union

import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image

from api.utils import load_model, index_to_url

class SemanticSearcher:
    """Object that performs semantic search on images and text

    Parameters
    ----------
    model_id : str
        HuggingFace model id for MultiModal model
    index : faiss.Index, optional
        Faiss index with embeddings to search, by default None
    """
    def __init__(self, model_id: str, index: faiss.Index=None, index_to_url: pd.DataFrame=None) -> None:
        self.model, self.processor = load_model(model_id)
        self.index = index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_to_url = index_to_url

    def process(self, batch: list[Union[Image.Image, str]]) -> np.array:
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
            processed_text = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            return self.model.get_text_features(**processed_text).detach().cpu().numpy()

    def __call__(self, query: list[Union[Image.Image, str]], k: int=5) -> list[str]:
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
        if not isinstance(query, list):
            query = [query]
        query_emb = self.process(query)
        query_emb /= np.linalg.norm(query_emb)
        # Getting Similarities
        _, I = self.index.search(query_emb, k)
        I = I.tolist()
        return [index_to_url(i, self.index_to_url.copy()) for i in I]

    @staticmethod
    def _infer_type(x: list[Union[Image.Image, str]]) -> str:
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