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
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI

import torch
import re
import base64
import os


SEARCH_NUM = 3
conversation_pairs = []
loaded_memory =""
dialogues = ""
query2=""

st.set_page_config(page_title = "book-chatbot")
st.title("책 추천 챗봇")


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

set_background("C:/Users/hhs87/Downloads/background_chat.png")




# 전역 메모리 초기화
if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = ConversationBufferMemory()

if 'cnt' not in st.session_state:
    st.session_state.cnt = 0

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

# csv load
loader = CSVLoader("C:/Users/hhs87/Downloads/updated_recommend_qa.csv", encoding="cp949")
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
    나의 질문에 문서에 기반해서 답을 하고, 문서와 관련이 없거나 모호하다면 문서에서 최대한 비슷한 정보를 찾아서 대답해. 
    이전 대화의 문맥이 주어진다면 문맥을 고려해서 대답해.
    문서에는 여러 책들이 존재하는데 그중에서 최대 3개까지만 골라서 추천하고 만약 다른 책을 알려달라고 하면 docs 안에 존재하는 다른 도서를 추천해줘.
    책 제목이 매우 유사하거나 띄어쓰기 혹은 맞춤법 차이인 경우는 유사한 책 중 옳바른 맞춤법을 가지는 도서 1개와 다른 도서를 말해줘.
    다른 책이나 다른 도서를 추천해 달라고 하면 문맥과 문서를 고려해서 문맥에 포함되지 않은 책 중 최대 3가지 까지 추천해줘. 
    절대 'AI:'을 사용하지마. 대답을 할 때 'AI:'을 포함하지마.
    최종 대답의 형식은 아래와 같이 해줘.
    해당 인기 도서로는 '책 제목'이 있습니다.

    이전 대화의 문맥 = {loaded_memory}
    문서 = {docs_input}
    질문 = {query_input}
'''
template3 ='''
            입력된 질문이 도서와 관련이 있다면 1을 출력하고 관련이 없다면 0을 출력해줘.
            질문 = {query_input} 
'''
template4 = '''
    너는 인간에게 도움이 되는 상냥한 도우미야.
    이전 대화의 문맥이 주어진다면 문맥을 고려해서 질문에 대답해.
    만약 다른 것을 알려달라고 한다면 이전 대화의 맥락을 고려해서 같은 주제의 다른 것을 말해줘.
    도서를 추천하지 말고 일상적인 대화를 해줘
    절대 'AI:'을 사용하지마. 대답을 할 때 'AI:'을 포함하지마. 

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
def load_chain2(_prompt) :
    chain2 = LLMChain(
        llm=OpenAIChat(),
        prompt=prompt2,verbose=True
    )
    return chain2
def load_chain3(_prompt) :
    chain3 = LLMChain(
        llm=OpenAIChat(),
        prompt=prompt3
    )
    return chain3
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



############################################



def count_occurrence(sentence):
    cnt = 1 if ('다른 책' or '다른 도서') in sentence else 0
    return cnt

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
 
if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    query = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')


if submitted and query:
    
    if run_chain3(prompt3, query) == '1' or "다른 책" in query or "다른 도서" in query:
        docs = db_search(query, SEARCH_NUM)
        docs = [doc.page_content for doc in docs]
        loaded_memory = st.session_state['conversation_memory'].load_memory_variables({})
        conversation_pairs.append(loaded_memory['history'])
        conversation_pairs=[conversation_pairs[-1]]

        if conversation_pairs[0] == 1:
            answer = run_chain(
            prompt,
            docs,
            query
            )
            print("1번이야")
            st.session_state.generated.append(answer)
        else:
            pattern = re.compile(r'Human: (.*?)(?=\nAI: |\Z)', re.DOTALL)
            matches = pattern.findall(conversation_pairs[-1])

            ai_pattern = re.compile(r'AI: (.*?)(?=\nHuman: |\Z)', re.DOTALL)
            ai_matches = ai_pattern.findall(conversation_pairs[-1])

            conversation_list = [{'Human': human.strip(), 'AI': ai.strip()} for human, ai in zip(matches, ai_matches)]

            if ("다른 책" in query or "다른 도서" in query) and conversation_pairs[0] != '' and st.session_state.cnt < 3:
                st.session_state.cnt += 1
                query2 = conversation_list[-st.session_state.cnt]['Human']
                docs = db_search(conversation_list[-st.session_state.cnt]['Human'], SEARCH_NUM) #[st.session_state.cnt:]
                docs = [doc.page_content for doc in docs]
                print("cnt:", st.session_state.cnt)
            else:
                docs = docs
                st.session_state.cnt = 0

            if st.session_state.cnt >= 3:
                answer = "죄송합니다. 해당 장르의 인기도서를 전부 추천했습니다. 다른 주제의 도서를 찾아주세요."
                st.session_state.cnt = 0
            
            else:
                while len(conversation_list) > 3:
                    conversation_list.pop(0)
                loaded_memory['history'] = "\n".join("Human: {}\nAI: {}".format(pair['Human'], pair['AI']) for pair in conversation_list)
                
                answer = run_chain2(
                    docs,
                    query,
                    loaded_memory
                )
                answer = answer.replace('AI:','')

            print("2번이야")
            st.session_state.generated.append(answer)

    else:
        loaded_memory = st.session_state['conversation_memory'].load_memory_variables({})
        conversation_pairs.append(loaded_memory['history'])
        conversation_pairs=[conversation_pairs[-1]]
        
        if conversation_pairs[0] == '':
            answer = OpenAIChat(max_tokens=200,temperature=0.1).predict(query)
        else:
            pattern = re.compile(r'Human: (.*?)(?=\nAI: |\Z)', re.DOTALL)
            matches = pattern.findall(conversation_pairs[-1])

            ai_pattern = re.compile(r'AI: (.*?)(?=\nHuman: |\Z)', re.DOTALL)
            ai_matches = ai_pattern.findall(conversation_pairs[-1])

            conversation_list = [{'Human': human.strip(), 'AI': ai.strip()} for human, ai in zip(matches, ai_matches)]

            while len(conversation_list) > 3:
                conversation_list.pop(0)
            #print(conversation_list)
            loaded_memory['history'] = "\n".join("Human: {}\nAI: {}".format(pair['Human'], pair['AI']) for pair in conversation_list)
                
            answer = run_chain4(
                prompt4,
                query,
                loaded_memory,
                
              )
            answer = answer.replace('AI:','')
            print("4번이야")

        st.session_state.generated.append(answer)
        
       
    if query2 =="":                
    # 질문과 대답 메모리에 저장
        st.session_state.past.append(query)
        st.session_state['conversation_memory'].save_context({"input": query}, {"output": answer})
    else:
        st.session_state.past.append(query)
        st.session_state['conversation_memory'].save_context({"input": query2}, {"output": answer})

loaded_memory = st.session_state['conversation_memory'].load_memory_variables({})

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))