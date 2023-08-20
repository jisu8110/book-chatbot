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

set_background('./images/background_theme.png')

# load data
book_df = pd.read_csv('./data/book.csv', encoding="utf-8")
book_df = book_df[['TITLE_NM', 'AUTHR_NM', 'PBLICTE_YEAR', 'KDC_NM']]

# filter data
book_tab1 = book_df[book_df["KDC_NM"] == "물리학"]
book_tab2 = book_df[book_df["KDC_NM"] == "소설"]
book_tab3 = book_df[book_df["KDC_NM"] == "양극지리"]
book_tab4 = book_df[book_df["KDC_NM"] == "경영관리"]


st.title("🔍 Theme")

tab1, tab2, tab3, tab4 = st.tabs(["⚖ 물리학", "📓 소설", "🌏 양극지리", "💰 경영관리"])

with tab1:
    st.write(book_tab1)
with tab2:
    st.write(book_tab2)
with tab3:
    st.write(book_tab3)
with tab4:
    st.write(book_tab4)
