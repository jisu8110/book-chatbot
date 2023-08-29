import streamlit as st
import pandas as pd
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./images/background_search.png')

st.title("🔍 Search")


col_a, col_b = st.columns([4,1])

with col_a:
    search_query = st.text_input(label='', placeholder="검색어를 입력해주세요.")

with col_b:
    st.write("")
    st.write("")
    if st.button('🔍'):
        pass 


st.write("")
st.write("")


col1, col2, col3, col4 = st.columns(4)

col1.radio('연령대', [
    '유아', 
    '10대',
    '20대',
    '30대',
    '40대',
    '50대',
    ], key='option1')


col2.radio('분야', [
    '전체', 
    '판타지',
    '소설',
    '로맨스',
    '지식',
    '자기계발',
    ], key='option2')

col3.radio('도서 종류', [
    '국내도서', 
    '외국도서',
    'eBook',
    '중고도서',
    'DVD',
    '티켓',
    ], key='option3')

col4.radio('정렬', [
    '추천순', 
    '인기순',
    '리뷰순',
    '조회순',
    '최신순',
    '이름순',
    ], key='option4')

