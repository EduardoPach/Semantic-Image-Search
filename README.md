# Image Semantic Search Project

![](assets/demo.gif)

This project uses OpenAI's CLIP model through Hugging Face's transformers library along with the Unsplash dataset (on data.csv file) and Facebook Faiss library to create an image semantic search engine.

The Faiss index was generated using a `compute_img_embeddings.py` script, which can be optionally used to generate the embeddings. In the repo, we already have the pre-computed embeddings, so you can directly use the pre-generated index files.

A Streamlit app was created in the `app.py` script to provide a web interface for searching the index using text queries. At the moment, the web interface only allows searching for images using text, but support for image-based queries will be added in the future.

A `SemanticSearcher` object was also created which is a wrapper around the CLIP model that allows using images or texts to query a Faiss index. The implementation of this object is in the `api` directory in the `SemanticSearcher.py` file.

## Usage 

### Using as it is

If you want the Image Semantic Search with the same sample data used you can follow these steps:

1. Clone the repository
2. Create the conda environment with `conda env create -f environment.yml`
3. Activate the environment with `conda activate <env_name>`
4. Run `streamlit run app.py` to start the web interface
5. Use the web interface to search for images by entering text queries. Image-based searches will be added in the future.


### Adapting to Different Datasets

If you want to adapt the current implementatio to a different dataset you'll have to pre-compute the embeddings. In order to do that use the `compute_img_embeddings.py` file, here's a bit more explanation on how to use it:

```python
python compute_img_embeddings.py \
    --model-id openai/clip-vit-base-patch32 \
    --batch-size 32 \
    --input-data /path/to/input.csv \
    --embeddings-temp-dir /path/to/temp_embeddings_dir \
    --embeddings-dir /path/to/embeddings_dir \
    --index-checkpoint index.faiss
```

Where:

- `--model-id` is the ID of the CLIP model to use from Hugging Face.
- `--batch-size` is the batch size to use when computing the embeddings for the images.
- `--input-data` is the path to the .csv file that contains a path column with the URL to the corresponding image.
- `--embeddings-temp-dir` is the directory name where temp embedding files will be stored during the job.
- `--embeddings-dir` is the final embedding directory where the concatenated embedding files will be stored.
- `--index-checkpoint` is the name of the existing FlatL2 faiss index to be used.


*Note:*  `--index-checkpoint` is optional since the repository already has the pre-computed embeddings.


If the data you're using is a directory of Images that are already stored on disk or somewhere else take a look on the `api/utils.py` file, especially in the `fetch_image` function.