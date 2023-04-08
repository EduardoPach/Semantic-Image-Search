import streamlit as st
from st_clickable_images import clickable_images

from utils import send_request, encode_image

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Image Semantic Search - Unsplash")

    with st.sidebar.container():
        c1, c2 = st.columns(2)
        mode = c1.radio("Query Mode", ["Text", "Image"])
        k = c2.slider("Number of Images", min_value=1, max_value=20, step=1, value=10)
        if mode == "Text":
            query = st.text_input("Query", value="Two dogs playing")
        else:
            query = None
            uploaded_image = st.file_uploader("Query", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                query = encode_image(uploaded_image.read()) 
                st.image(uploaded_image, use_column_width=True, caption="Query Image")


    if not query:
        return 
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