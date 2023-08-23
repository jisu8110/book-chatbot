import streamlit as st
from streamlit_chat import message

from peft import PeftModel, PeftConfig
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI

import torch
import re
import base64
import os


# background image
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

set_background('./images/background_chat.png')



SEARCH_NUM = 3
loaded_memory =""
dialogues = ""

# 전역 메모리 초기화
if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = ConversationBufferMemory()


with st.sidebar:
    openai_api_key = st.text_input(label='#### Your OpenAI API Key', placeholder="OpenAI api key를 입력하세요.", type="password")
    load_data = st.button('Enter')

    if load_data:
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # csv load
        loader = CSVLoader('./data/chatbot_prompts_v6.csv', encoding="utf-8")
        data = loader.load()

        # text split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
        data_split = text_splitter.split_documents(data)

        # embedding
        embeddings = OpenAIEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        db = FAISS.from_documents(data_split, embeddings)

docs_memory = []
cnt=0
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(그만):])).item():
                return True

        return False

def gen_booktest(x):
    prompts = f"### 응답: {x}\n\n### 유형:"
    gened = model.generate(
        **tokenizer(
            prompts,
            return_tensors='pt',
            return_token_type_ids=False
        ).to(0),
        max_new_tokens=1,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    return 1 if (tokenizer.decode(gened[0]).replace(prompts+ " ", "")=="도서") else 0

# prompt template
template = '''
    너는 책에 대해서 답해주는 도서 챗봇이야.
    나의 질문에 문서에 기반해서 답을 하고, 문서와 관련이 없거나 모호하다면 너가 생각해서 대답해.

    문서 = {docs_input}
    질문 = {query_input}

'''
template2 = '''
    너는 책에 대해서 답해주는 도서 챗봇이야.
    나의 질문에 문서에 기반해서 답을 하고, 문서와 관련이 없거나 모호하다면 너가 생각해서 대답해.
    이전 대화의 문맥이 주어진다면 문맥을 고려해서 대답해.
    책은 한 번에 최대 3개까지 추천하고 만약 다른 책을 알려달라고 하면 docs 안에 존재하는 다른 도서를 추천해줘.

    이전 대화의 문맥 = {loaded_memory}
    문서 = {docs_input}
    질문 = {query_input}
'''
template3 ="""
            입력된 질문이 도서와 관련이 있다면 1을 출력하고 관련이 없다면 0을 출력해줘.
            질문 = {query_input} 
"""
template4 = '''
    너는 인간에게 도움이 되는 상냥한 assistant야.
    이전 대화의 문맥이 주어진다면 문맥을 고려해서 질문에 대답해.
    만약 다른 것을 알려달라고 한다면 이전 대화의 맥락을 고려해서 같은 주제의 다른 것을 말해줘.
    bot: 이나 AI:은 각 대화마다 최대 1번만 사용해.

    이전 대화의 문맥 = {loaded_memory}

    질문 = {query_input}
'''


# prompt
prompt = PromptTemplate(
    input_variables=[
        "docs_input", 
        "query_input",
    ],
    template=template,
)

prompt2 = PromptTemplate(
    input_variables=[
        "docs_input", 
        "query_input",
        "loaded_memory"
    ],
    template=template2,
)

prompt3 = PromptTemplate(
    input_variables=[
        "query_input",
    ],
    template=template3,
)
 
prompt4 = PromptTemplate(
    input_variables=[
        "query_input",
        "loaded_memory"
    ],
    template=template4,
)


# search similarity
def db_search(query: str, k: int):
    docs = db.similarity_search(query, k)
    return docs

@st.cache_resource
def load_local_llm():
    #lora 모델 가져오기
    
    peft_model_id = "hhs8746/book_test"
    config = PeftConfig.from_pretrained(peft_model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, peft_model_id)
    return model

model = OpenAI() #load_local_llm()
llm = model

def gen(x):
    gened = model.generate(
        **tokenizer(
            f"### 질문: {x}\n\n### 답변:",
            return_tensors='pt',
            return_token_type_ids=False
        ).to(0),
        max_new_tokens=100,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
        stopping_criteria=stopping_criteria
    )
    print(tokenizer.decode(gened[0]))

def generate_prompt_from_docs(query: str, k: int):
    docs = db.similarity_search(query, k)
    # Extract page_content from the first k docs
    doc_strings = [doc.page_content.split('\nAnswer: ')[1] for doc in docs[:k]]

    # Format the documents into a single string with a separator
    docs_input = " | ".join(doc_strings)

    # Construct the prompt using the given template
    prompt = f"""
    너는 책에 대해서 답해주는 도서 챗봇이야.
    나의 질문에 문서에 기반해서 답을 하고, 문서와 관련이 없거나 모호한 질문은 모른다고 대답해.

    문서 = {docs_input}
    질문 = {query}
    """

    return prompt


# load chain
@st.cache_resource
def load_chain(_prompt) :
    chain = LLMChain(
        llm=OpenAIChat(),
        prompt=prompt
    )
    return chain

@st.cache_resource
def load_chain2(_prompt) :
    chain2 = LLMChain(
        llm=OpenAIChat(),
        prompt=prompt2,verbose=True
    )
    return chain2

@st.cache_resource
def load_chain3(_prompt) :
    chain3 = LLMChain(
        llm=OpenAIChat(),
        prompt=prompt3
    )
    return chain3

@st.cache_resource
def load_chain4(_prompt) :
    chain4 = LLMChain(
        llm=OpenAIChat(temperature=0.2),
        prompt=prompt4,verbose=True
    )
    return chain4

#run chain
def run_chain(prompt, docs,query: str):
    chain = load_chain(prompt)
    response = chain.run(
    docs_input=docs,
    query_input=query,
    )
    return response

def run_chain2(docs, query: str,loaded_memory):
    global prompt2
    chain2 = load_chain2(prompt2)
    response = chain2.run(
    docs_input=docs,
    query_input=query,
    loaded_memory = loaded_memory['history']
    )
    return response

def run_chain3(prompt3,query: str):
    chain3 = load_chain3(prompt3)
    response = chain3.run(
    query_input=query,
    )
    return response

def run_chain4(prompt4,query: str,loaded_memory):
    chain4 = load_chain4(prompt4)
    response = chain4.run(
    query_input=query,
    loaded_memory = loaded_memory['history']
    )
    return response



# 대화 기록 불러오기
############################################


def count_occurrence(sentence):
    cnt = 1 if ('다른 책' or '다른 도서') in sentence else 0
    return cnt

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
 
if 'past' not in st.session_state:
    st.session_state['past'] = []


st.title("책 추천 챗봇")

with st.form('form', clear_on_submit=True):
    query = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and query:
    if run_chain3(prompt3, query) == '1' or "다른 책" in query:
        docs = db_search(query, SEARCH_NUM)
        if len(st.session_state['conversation_memory'].load_memory_variables({})) ==0:
            answer = run_chain(
            prompt,
            docs,
            query
            )
            print("1번이야")
            st.session_state.generated.append(answer)
        else:
            loaded_memory = st.session_state['conversation_memory'].load_memory_variables({})
            pattern = re.compile(r'Human: (.*?)\nAI: (.*?)\n', re.DOTALL)
            matches = pattern.findall(loaded_memory['history'])

            conversation_pairs = []

            for match in matches:
                human, ai = match
                pair = {'Human': human, 'AI': ai}
                conversation_pairs.append(pair)
            if len(conversation_pairs) >=3:
                conversation_pairs.pop(0)
                loaded_memory['history'] = ""
                for pair in conversation_pairs:
                    loaded_memory['history'] += "Human: {}\nAI: {}\n".format(pair['Human'], pair['AI']) 
            
            answer = run_chain2(
            query,
            docs,
            loaded_memory
            )
            print("2번이야")
            st.session_state.generated.append(answer)
            
    else:
        if len(st.session_state['conversation_memory'].load_memory_variables({})) ==0:
            answer = OpenAIChat(max_tokens=200,temperature=0.2).predict(query)
            
        else:
            loaded_memory = st.session_state['conversation_memory'].load_memory_variables({})
            pattern = re.compile(r'Human: (.*?)\nAI: (.*?)\n', re.DOTALL)
            matches = pattern.findall(loaded_memory['history'])

            conversation_pairs = []

            for match in matches:
                human, ai = match
                pair = {'Human': human, 'AI': ai}
                conversation_pairs.append(pair)
            if len(conversation_pairs) >=3:
                conversation_pairs.pop(0)
                loaded_memory['history'] = ""
                for pair in conversation_pairs:
                    loaded_memory['history'] += "Human: {}\nAI: {}\n".format(pair['Human'], pair['AI']) 

            # print(conversation_pairs)

            answer = run_chain4(
                prompt4,
                query,
                loaded_memory,
                
              )
            print("4번이야")
        st.session_state.generated.append(answer)
        
        #answer = OpenAIChat(max_tokens=200,temperature=0.2).predict(query)
                
    # 질문과 대답 메모리에 저장
    st.session_state.past.append(query)
    st.session_state['conversation_memory'].save_context({"input": query}, {"output": answer})

# 대화 기록 불러오기 및 출력
loaded_memory = st.session_state['conversation_memory'].load_memory_variables({})



if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

