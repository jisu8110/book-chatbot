import streamlit as st
from PIL import Image


# main
st.image("./images/logo2.png", width=240)

st.markdown('''
<h2><span style="color: #D0FD5C;">Book-Chat</span> 채팅에 오신 것을 환영합니다!</h2>
''', unsafe_allow_html=True)
st.markdown("나에게 딱 맞는 책을 찾아보세요!")

st.markdown('<a href="/chat" target="_self">👉 **Book-Chat과 대화해보기**</a>', unsafe_allow_html=True)


# linktree
st.markdown(
    """
    <style>
    .link-card {
        display: flex;
        flex-direction: column;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #D0FD5C;
        border-radius: 25px;
        background-color: white; 
        box-shadow: 0px 0px 5px #F4EDEC;
    }
    .link-card:hover {
    background-color: #D0FD5C;
    }
    a {
        color: black!important;
        text-decoration: none!important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def create_link_card(title, url):
    container = st.container()
    container.markdown(
        f'<div class="link-card"><a href="{url}" target="_blank">{title}</a></div>',
        unsafe_allow_html=True,
    )
    return container


# 책 추천
link_container = st.container()
with link_container:
    col1, col2 = st.columns(2)
    with col1:
        st.image("./images/book_image1.jpg", width=100)
        create_link_card(
            "🍏 인기 1순위 : 세이노의 가르침",
            "https://cafe.naver.com/bitamin123/2512",
        )
    with col2:
        st.image("./images/book_image2.jpg", width=100)
        create_link_card(
            "🍈 신작 : 역행자 확장판",
            "https://blog.naver.com/bita_min",
        )

# 채팅/랭크 페이지 이동
create_link_card(
    "💘 내 취향을 저격하는 책은 뭘까?",
    "/chat",
)
create_link_card(
    "🔥 요즘 제일 핫한 인기 책을 알려줘",
    "/rank",
)