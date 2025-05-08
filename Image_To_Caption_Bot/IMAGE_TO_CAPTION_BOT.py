import requests
import streamlit as st
col1, col2, col3 = st.columns([3, 12, 3])
with col1:
    st.write("")
with col2:
    st.title("UPLOAD TO CAPTION")
with col3:
    st.write("")
API_URL = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
headers = {"Authorization": "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"}

def query(filename):
    data = filename.read()
    response = requests.post(API_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return None

uploaded_file = st.file_uploader("Enter your IMAGE", type="jpg")

if st.button("Generate Caption"):
    output = query(uploaded_file)
    st.write(output)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    