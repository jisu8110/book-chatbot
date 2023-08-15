import streamlit as st
from PIL import Image

# profile
image = Image.open('./images/profile_image.jpg')
name = "수지"
description = "안녕하세요! 수지에요"

st.set_page_config(layout="wide")

st.title('Book-ChatBot')
st.text('챗봇 만드는 중입니다')

# Sidebar setup
st.sidebar.image(image, caption=None, width=100)  # 프로필 이미지 표시
st.sidebar.markdown(f"## {name}")  
st.sidebar.markdown(description)  