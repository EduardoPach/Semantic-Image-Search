# Image Semantic Search Project

This project uses OpenAI's CLIP model through Hugging Face's transformers library along with the Unsplash dataset (on data.csv file) and Facebook Faiss library to create an image semantic search engine.

The Faiss index was generated using a `compute_img_embeddings.py` script, which can be optionally used to generate the embeddings. In the repo, we already have the pre-computed embeddings, so you can directly use the pre-generated index files.

A Streamlit app was created in the `app.py` script to provide a web interface for searching the index using text queries. At the moment, the web interface only allows searching for images using text, but support for image-based queries will be added in the future.

A `SemanticSearcher` object was also created which is a wrapper around the CLIP model that allows using images or texts to query a Faiss index. The implementation of this object is in the `api` directory in the `SemanticSearcher.py` file.

## Usage

1. Clone the repository
2. Create the conda environment with `conda env create -f environment.yml`
3. Activate the environment with `conda activate <env_name>`
4. Optionally, run `python compute_img_embeddings.py` to generate the Faiss index and csv files
5. Run `streamlit run app.py` to start the web interface
6. Use the web interface to search for images by entering text queries. Image-based searches will be added in the future.
