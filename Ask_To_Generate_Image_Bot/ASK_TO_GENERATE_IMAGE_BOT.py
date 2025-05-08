import requests
import streamlit as st
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")
with col2:
    st.title("ASK TO GENERATE IMAGE")
with col3:
    st.write("")
API_URL = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
headers = {"Authorization": "yyyyyyyyyyyyyyyyyyyyyyyyyyyy"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs":st.text_input("Enter your Imagination"),
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
if st.button("CLICK TO GENERATE IMAGE"):
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image)