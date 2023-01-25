from __future__ import annotations

import os
import shutil
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from api.utils import ImageBatchGenerator, compute_embedding

def main(args):
    embeddings_temp_dir_path = args.embeddings_temp_dir
    embeddings_dir_path = args.embeddings_dir
    embedding_file_name = args.embedding_filename
    data_file = args.input_data
    batch_size = args.batch_size

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = pd.read_csv(data_file)
    urls = data["path"].tolist()
    batch_generator = ImageBatchGenerator(urls, batch_size)

    loop = tqdm(batch_generator)
    embeddings_temp_dir = Path(embeddings_temp_dir_path)
    embeddings_dir = Path(embeddings_dir_path)
    embedding_file_path = embeddings_dir / (embedding_file_name+'.npy')
    url_file_path = embeddings_dir / (embedding_file_name+'.csv')
    for batch_idx, batch in enumerate(loop):
        images, urls = batch.values()
        embedding = compute_embedding(model, processor, images, device=device)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
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
        "--embedding-filename",
        type=str,
        default="embeddings",
        help="Name of the final embedding file to be stored inside embeddings_dir."
    )
    
    args = parser.parse_args()
    main(args)