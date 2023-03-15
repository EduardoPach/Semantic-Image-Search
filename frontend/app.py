import streamlit as st
from st_clickable_images import clickable_images

from utils import send_request

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Image Semantic Search - Unsplash")
    c1, c2 = st.columns(2)
    query = c1.text_input("Image Query", value="Two dogs playing")
    k = c2.slider("Number of Images", min_value=1, max_value=20, step=1, value=10)
    img_urls = send_request(query, k)
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