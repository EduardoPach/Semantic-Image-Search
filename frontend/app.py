import streamlit as st
from st_clickable_images import clickable_images

import utils

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Image Semantic Search - Unsplash")

    query, k = utils.start_sidebar()

    if not query:
        return 
    
    st.subheader("Query Results")
    img_urls = utils.send_request(query, k)
    clicked = clickable_images(
        img_urls, 
        img_style={"margin": "5px", "height": "200px"},
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
        }
    )

    if clicked==-1:
        return
    
    st.subheader("Refined Results")
    refined_img_urls = utils.refined_search(img_urls[clicked], k)
    clickable_images(
        refined_img_urls, 
        img_style={"margin": "5px", "height": "200px"},
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
        }
    )
    
if __name__=="__main__":
    main()