import streamlit as st
# from streamlit_modal import Modal
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
    [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);
    }
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./images/background_main.png')


st.markdown(
    """
    <style>
    .link-card {
        display: flex;
        flex-direction: column;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid #62C8DF;
        border-radius: 25px;
        background-color: white; 
        box-shadow: 0px 0px 5px #B0C6CB;
    }
    .link-card:hover {
        background-color: #62C8DF;
    }
    .chat-link {
        padding: 10px 20px;
        border: 2px solid transparent;
        border-radius: 10px;
        background-color: #70E5FF;
        box-shadow: 0px 0px 5px #B0C6CB;
    }
    .chat-link:hover {
        background-color: transparent;
        border-color: #70E5FF;
        color: #D3E631;
    }
    a {
        color: black!important;
        text-decoration: none!important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# main
# st.image("./images/logo2.png", width=240)

st.markdown('''
<h2><span style="color: #62C8DF;">Book-Chat</span> 채팅에 오신 것을 환영합니다!</h2>
''', unsafe_allow_html=True)


st.markdown("북챗과 함께 나에게 딱 맞는 책을 찾아보세요!")
st.markdown('<h4><a href="/chat" target="_self" class="chat-link">🚀 Book-Chat과 대화해보기</a></h4>', unsafe_allow_html=True)


st.write("")
st.write("")
st.write("")


def create_link_card(title, url):
    container = st.container()
    container.markdown(
        f'<div class="link-card"><a href="{url}" target="_blank">{title}</a></div>',
        unsafe_allow_html=True,
    )

# def create_modal_card(theme, title, desc):
#     open_modal = st.button(label=theme+title)
#     if open_modal:
#         with Modal(key='key', title=title).container():
#             st.markdown(desc)


# 책 추천
link2_container = st.container()
with link2_container:
    col3, col4 = st.columns(2)
    with col3:
        st.image("./images/book1.png")
        create_link_card(
            "🐳 인기 1순위 : 1%를 읽는 힘",
            'https://www.yes24.com/Product/Goods/121812955',
            # "국내 최고의 자본시장 분석가이자, 경제·주식 분야 파워 인플루언서로 타의 추종을 불허하는 독보적인 시각을 제시하는 메르의 모든 투자 노하우를 담은 책이다."
        )
    with col4:
        st.image("./images/book2.png")
        create_link_card(
            "🦄 신작 : 메리골드 마음 세탁소",
            'https://www.yes24.com/Product/Goods/117716170',
            # "『메리골드 마음 세탁소』는 한밤중 언덕 위에 생겨난, 조금 수상하고도 신비로운 세탁소에서 벌어지는 일들을 그린 힐링 판타지 소설이다."
        )

# 페이지 이동
create_link_card(
    "💘 내 취향을 저격하는 책은 뭘까?",
    "/chat",
)
create_link_card(
    "🔥 요즘 제일 핫한 인기 책을 알려줘",
    "/rank",
)
create_link_card(
    "🧙 판타지 장르를 보고 싶어!",
    "/theme",
)