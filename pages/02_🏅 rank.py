import streamlit as st
import pandas as pd

from langchain.document_loaders.csv_loader import CSVLoader

st.title("Ranking")

# load data
book_df = pd.read_csv('./data/book.csv', encoding="utf-8")

# sort data
sorted_book = book_df.sort_values('PBLICTE_YEAR', ascending=False)

tab1, tab2= st.tabs(["최신 도서 순", "검색 순"])

with tab1:
    st.write(sorted_book[['TITLE_NM', 'AUTHR_NM', 'PBLICTE_YEAR', 'KDC_NM']])

with tab2:
    st.write("검색검색")


