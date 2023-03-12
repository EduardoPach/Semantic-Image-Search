from __future__ import annotations

import os
import shutil
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from api import ImageBatchGenerator, SemanticSearcher

def main(
    input_dir: str,
    model_id: str="openai/clip-vit-base-patch32",
    batch_size: int=100
) -> None:
    """Precompute image embeddings and save them to disk with 
    a faiss index object and a index to image path table

    Parameters
    ----------
    input_dir : str
        A directory containing images to process
    model_id : str, optional
        Hugging Face model id from pretrained CLIP model to use
        , by default "openai/clip-vit-large-patch14"        
    batch_size : int, optional
        Batch size to use while computing embeddings, by default 100
    """
    embeddings_temp_dir_path = "temp_embeddings"
    embeddings_dir_path = "embeddings"
    index_checkpoint = "00001.index"
    db_checkpoint = "00001.parquet"
    print("----------- Initializing Semantic Searcher -----------")
    searcher = SemanticSearcher(model_id=model_id)
    embedding_dimension = searcher.model.visual_projection.out_features

    dir_ = Path(input_dir)
    img_list = list(dir_.glob("*"))
    img_list = [str(img) for img in img_list]
    batch_generator = ImageBatchGenerator(img_list, batch_size)

    if os.path.exists(index_checkpoint):
        index = faiss.read_index(index_checkpoint)
    else:
        index = faiss.IndexFlatL2(embedding_dimension)

    loop = tqdm(batch_generator)
    embeddings_temp_dir = Path(embeddings_temp_dir_path)
    embeddings_dir = Path(embeddings_dir_path)

    for batch_idx, batch in enumerate(loop):
        images, img_path = batch.values()
        embedding = searcher.process(images)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        if not embeddings_temp_dir.exists():
            os.mkdir(embeddings_temp_dir)
        name_prefix = str(embeddings_temp_dir / f"{batch_idx:05d}")
        pd.DataFrame(img_path, columns=["path"]).to_parquet(f"{name_prefix}.parquet", index=False)
        np.save(f"{name_prefix}.npy", embedding)

    url_list = [pd.read_parquet(url_file) for url_file in sorted(embeddings_temp_dir.glob("*.parquet"))]
    embedding_list = [np.load(embedding_file) for embedding_file in sorted(embeddings_temp_dir.glob("*.npy"))]
    embedding_full = np.concatenate(embedding_list)
    img_path = pd.concat(url_list).reset_index(drop=True)

    if not embeddings_dir.exists():
        os.mkdir(embeddings_dir)

    index.add(embedding_full)
    faiss.write_index(index, str(embeddings_dir / index_checkpoint))

    if os.path.exists(index_checkpoint):
        curr_img_path = pd.read_csv(db_checkpoint)
        img_path = pd.concat([curr_img_path, img_path]).reset_index(drop=True)

    img_path.to_parquet(str(embeddings_dir / db_checkpoint), index=False)

    shutil.rmtree(embeddings_temp_dir)

if __name__ == "__main__":
    main(input_dir="test_dir")