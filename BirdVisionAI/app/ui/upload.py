"""ui/upload.py
Image upload component.
"""
import streamlit as st
from PIL import Image

from ui.templates import section_label


def upload_image() -> Image.Image | None:
    """Styled file uploader with live camera support — returns a Pillow Image in RGB mode, or None."""
    st.markdown(section_label("Select Input Method"), unsafe_allow_html=True)
    tab_file, tab_cam = st.tabs(["📁 Upload Photo", "📸 Live Camera"])

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload a bird photo",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="bird_file_uploader",
        )

    with tab_cam:
        captured_cam = st.camera_input(
            "Take a live photo of a bird",
            label_visibility="collapsed",
            key="bird_camera_input",
        )

    # Prioritise camera if taken, otherwise uploaded file
    if captured_cam is not None:
        return Image.open(captured_cam).convert("RGB")
    elif uploaded_file is not None:
        return Image.open(uploaded_file).convert("RGB")
    return None


if __name__ == "__main__":
    st.set_page_config(page_title="Upload Test")
    st.title("Bird Species Classifier")
    img = upload_image()
    if img:
        st.image(img, caption="Loaded Image", use_container_width=True)
        st.success("Image successfully loaded!")
    else:
        st.info("Please upload an image or take a photo to begin.")

