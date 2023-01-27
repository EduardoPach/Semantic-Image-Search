import torch
import streamlit as st
from st_clickable_images import clickable_images
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from api.utils import search_image_from_text

@st.experimental_singleton
def load_model() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Image Semantic Search - Unsplash")
    model, processor = load_model()
    c1, c2 = st.columns(2)
    query = c1.text_input("Image Query", value="Two dogs playing")
    k = c2.slider("Number of Images", min_value=1, max_value=20, step=1)
    img_urls = search_image_from_text(model, processor, query, k)
    clicked = clickable_images(img_urls, img_style={"margin": "5px", "height": "200px"})

if __name__=="__main__":
    main()