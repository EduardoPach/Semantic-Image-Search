from __future__ import annotations

import os
import shutil
import argparse
from pathlib import Path

import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from api import ImageBatchGenerator, SemanticSearcher

def main(args: argparse.Namespace) -> None:
    """Computes the normalized embeddings for a list of Image URLs
    and store them in .index file with a correspondet .csv file that 
    matches indexes. 

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments
    """
    embeddings_temp_dir_path = args.embeddings_temp_dir
    embeddings_dir_path = args.embeddings_dir
    index_checkpoint = args.index_checkpoint
    data_file = args.input_data
    batch_size = args.batch_size
    model_id = args.model_id
    print("----------- Initializing Semantic Searcher -----------")
    searcher = SemanticSearcher(model_id=model_id)
    embedding_dimension = searcher.model.visual_projection.out_features

    data = pd.read_csv(data_file)
    urls = data["path"].tolist()
    batch_generator = ImageBatchGenerator(urls, batch_size)

    if index_checkpoint:
        index = faiss.read_index(index_checkpoint)
    else:
        index = faiss.IndexFlatL2(embedding_dimension)

    loop = tqdm(batch_generator)
    embeddings_temp_dir = Path(embeddings_temp_dir_path)
    embeddings_dir = Path(embeddings_dir_path)
    for batch_idx, batch in enumerate(loop):
        images, urls = batch.values()
        embedding = searcher.process(images)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        index.add(embedding)
        url_names = embeddings_temp_dir / f"{batch_idx:05d}.csv"
        if not embeddings_temp_dir.exists():
            os.mkdir(embeddings_temp_dir)
        pd.DataFrame(urls, columns=["url"]).to_csv(url_names, index=False)
    url_list = [pd.read_csv(url_file) for url_file in sorted(embeddings_temp_dir.glob("*.csv"))]
    urls = pd.concat(url_list).reset_index(drop=True)
    if not embeddings_dir.exists():
        os.mkdir(embeddings_dir)
    if index_checkpoint:
        url_file_path = embeddings_dir / (index_checkpoint.split(".")[0]+'.csv')
        curr_urls = pd.read_csv(url_file_path)
        urls = pd.concat([curr_urls, urls]).reset_index(drop=True)
        urls.to_csv(url_file_path, index=False)
        faiss.write_index(index, str(embeddings_dir / index_checkpoint))
    else:
        n = len(list(embeddings_dir.glob(".csv"))) + 1 # Number of .csv (metadata linked to a index) n_csv = n_index
        url_file_path = embeddings_dir / f"{n:05d}.csv"
        index_file_path = embeddings_dir / f"{n:05d}.index"
        urls.to_csv(url_file_path, index=False)
        faiss.write_index(index, str(index_file_path))
    shutil.rmtree(embeddings_temp_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Which version of CLIP to use"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size when computing the Embeddings for the images."
    )

    parser.add_argument(
        "--input-data",
        type=str,
        help="Path to .csv file that contains a path column with the URL to the correspondent image."
    )

    parser.add_argument(
        "--embeddings-temp-dir",
        type=str,
        default="./temp_embeddings",
        help="Directory name where temp embedding files will be stored during job."
    )

    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="./embeddings",
        help="Final embedding directory where concat embedding files will be stored."
    )

    parser.add_argument(
        "--index-checkpoint",
        type=str,
        default="",
        help="Name of existing FlatL2 faiss index to be used."
    )
    
    args = parser.parse_args()
    main(args)