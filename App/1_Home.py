import streamlit as st
from test import get_mask
import io
def pil_to_bytes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')  # Adjust the format as needed (PNG, JPEG, etc.)
    return img_bytes.getvalue()

st.title("Carvana's cars Background Remover")
img = st.file_uploader("Upload your image :", type=["jpg", "jpeg", "png"])

rd = st.radio("Model options :", ["Pre-trained on ImgaeNet", "Not pre-trained"], horizontal=True)
rd2 = st.radio("Type of the result :", ["Mask", "Cropped", "Highlighted"], horizontal=True)

col1, col2 = st.columns(2)

col1.write("Original Image")
col2.write("Result")
if img:
    col1.image(img, use_column_width=True)
    col2.image(get_mask(img, rd, rd2), use_column_width=True)
    st.download_button(
            label="Download the resulting image",
            data= pil_to_bytes(get_mask(img, rd, rd2)),
            file_name="img.png"
          )

