import streamlit as st

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

import os
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

SEARCH_NUM = 4

# csv load
loader = CSVLoader('./data/chatbot_prompts_v6.csv', encoding="utf-8")
data = loader.load()

# text split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
data_split = text_splitter.split_documents(data)

# embedding
embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")
db = FAISS.from_documents(data_split, embeddings)

# prompt template
template = '''
    너는 책에 대해서 답해주는 도서 챗봇이야.
    나의 질문에 문서에 기반해서 답을 하고, 문서와 관련이 없거나 모호한 질문은 모른다고 대답해.

    문서 = {docs_input}
    질문 = {query_input}

    모든 대답의 첫 마디에 '안녕!'이라고 덧붙여서 말해.
'''

# prompt
prompt = PromptTemplate(
    input_variables=[
        "docs_input", 
        "query_input"
    ],
    template=template,
)

# search similarity
def db_search(query: str, k: int):
    docs = db.similarity_search(query, k)
    return docs

# load chain
@st.cache_resource
def load_chain(_prompt) :
    chain = LLMChain(
        llm=OpenAI(),
        prompt=prompt
    )
    return chain

# run chain
def run_chain(prompt, docs, query: str):
    chain = load_chain(prompt)
    response = chain.run(
        docs_input=docs,
        query_input=query
    )
    return response

############################################

def main() :
    st.set_page_config(page_title = "book-chatbot")
    st.title("책 추천 챗봇")

    query = st.text_input("질문을 입력하세요: ") 

    if query:
        docs = db_search(query, SEARCH_NUM)
        response = run_chain(
            prompt,
            docs,
            query
        )
        
        st.write("챗봇: ")
        st.write(response)  


if __name__ == "__main__" :
    main()