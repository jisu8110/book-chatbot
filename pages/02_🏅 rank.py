import streamlit as st
import pandas as pd
import base64

# from langchain.document_loaders.csv_loader import CSVLoader

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

set_background('./images/background_rank.png')


st.title("🏅 Ranking")

# load data
book_df = pd.read_csv('./data/book.csv', encoding="utf-8")

# sort 최신 도서 순
sorted_book1 = book_df.sort_values('PBLICTE_YEAR', ascending=False)

# sort 도서명 순
# sorted_book2 = book_df.sort_values('TITLE_NM', ascending=False)


col1, col2 = st.columns([4,1])

with col1:
    search_query = st.text_input(label='', placeholder="검색어를 입력해주세요.")

with col2:
    st.write("")
    st.write("")
    if st.button('🔍'):
        pass 


st.write("")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🆕 최신 도서 순", "👍 대출 인기 순", "🔍 검색 인기 순", "💕 베스트셀러", "🥰 스테디셀러"])

with tab1:
    st.write(sorted_book1[['TITLE_NM', 'AUTHR_NM', 'PBLICTE_YEAR', 'KDC_NM']])

# with tab2:
#     st.write(sorted_book2[['TITLE_NM', 'AUTHR_NM', 'PBLICTE_YEAR', 'KDC_NM']])


