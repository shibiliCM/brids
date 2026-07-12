"""ui/upload.py
Image upload component.
"""
import streamlit as st
from PIL import Image

from ui.templates import section_label


def upload_image() -> Image.Image | None:
    """Styled file uploader — returns a Pillow Image in RGB mode, or None."""
    st.markdown(section_label("Upload a bird photo"), unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        return Image.open(uploaded).convert("RGB")
    return None


if __name__ == "__main__":
    st.set_page_config(page_title="Upload Test")
    st.title("Bird Species Classifier")
    img = upload_image()
    if img:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.success("Image successfully uploaded!")
    else:
        st.info("Please upload an image to begin.")
