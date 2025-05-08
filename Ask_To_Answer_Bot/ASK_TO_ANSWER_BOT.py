import streamlit as st
import google.generativeai as genai
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")
with col2:
    st.title("ASK TO ANSWER")
with col3:
    st.write("")
genai.configure(api_key="xxxxxxxxxxxxxxxxxxxxxxx")
text = st.text_input("Enter your Query:")
col1, col2, col3 = st.columns([18, 12, 18])
with col1:
    st.write("")
with col2:
    get_answer = st.button("GET ANSWER")
with col3:
    st.write("")
model = genai.GenerativeModel('gemini-pro')   
chat = model.start_chat(history=[])
if get_answer and text.strip():
    response = chat.send_message(text)
    if response.candidates:
        content = response.candidates[0].content.parts[0].text
        st.markdown(f"### Response:\n{content}")
else:
    st.error("Enter Your Query in the above Text Box")