import faiss
import pandas as pd
import streamlit as st
from st_clickable_images import clickable_images

from api import SemanticSearcher, index_to_url

@st.experimental_singleton
def load_searcher() -> SemanticSearcher:
    index = faiss.read_index("embeddings/00001.index")
    searcher = SemanticSearcher("openai/clip-vit-base-patch32", index)
    return searcher

@st.cache
def load_url2index() -> pd.DataFrame:
    url2index = pd.read_csv("embeddings/00001.csv")
    return url2index

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Image Semantic Search - Unsplash")
    searcher = load_searcher()
    db = load_url2index()
    c1, c2 = st.columns(2)
    query = c1.text_input("Image Query", value="Two dogs playing")
    k = c2.slider("Number of Images", min_value=1, max_value=20, step=1, value=10)
    I = searcher(query, k=k)
    img_urls = index_to_url(I, db)
    clicked = clickable_images(
        img_urls, 
        img_style={"margin": "5px", "height": "200px"},
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
        }
    )

if __name__=="__main__":
    main()