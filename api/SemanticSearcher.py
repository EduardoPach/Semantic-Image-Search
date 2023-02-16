from __future__ import annotations

from typing import Union

import torch
import faiss
import numpy as np
from PIL import Image

from api.utils import load_model

class SemanticSearcher:
    """_summary_

    Parameters
    ----------
    model_id : str
        _description_
    index : faiss.Index, optional
        _description_, by default None
    """
    def __init__(self, model_id: str, index: faiss.Index=None) -> None:
        self.model, self.processor = load_model(model_id)
        self.index = index
        self.device = "cuda" if torch.cuda.is_available else "cpu"

    def __call__(self, batch: Union[list[Image.Image], list[str]]) -> np.array:
        """_summary_

        Parameters
        ----------
        batch : Union[list[Image.Image], list[str]]
            _description_

        Returns
        -------
        np.array
            _description_
        """
        self.model.to(self.device)
        mode = self._infer_type(batch)
        if mode=="visual":
            processed_images = self.processor(images=batch, return_tensors="pt").to(self.device)
            return self.model.get_image_features(**processed_images).detach().cpu().numpy()
        elif mode=="text":
            processed_text = self.processor(text=batch, return_tensors="pt").to(self.device)
            return self.model.get_text_features(**processed_text).detach().cpu().numpy()

    def search(self, query: Union[list[Image.Image], list[str]], k: int=5) -> np.array:
        """_summary_

        Parameters
        ----------
        query : Union[list[Image.Image], list[str]]
            _description_
        k : int, optional
            _description_, by default 5

        Returns
        -------
        np.array
            _description_
        """
        # Query embedding
        query_emb = self([query])
        query_emb /= np.linalg.norm(query_emb)
        # Getting Similarities
        _, I = self.index(query_emb, k)
        
        return I

    @staticmethod
    def _infer_type(x: Union[list[Image.Image], list[str]]) -> str:
        return "visual" if isinstance(x[0], Image.Image) else "text"