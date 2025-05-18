import streamlit as st
from huggingface_hub import InferenceClient
import os

# 토큰 불러오기 (필수)
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Hugging Face Inference API 클라이언트
client = InferenceClient(
    model="tiiuae/falcon-rw-1b",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
st.set_page_config(page_title="🐍 파이썬 코드 질문 챗봇", layout="wide")
st.title("💬 파이썬 프로그래밍 도우미 (Hugging Face 무료 API)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자 입력
user_prompt = st.chat_input("파이썬에 대해 궁금한 걸 물어보세요!")

# 대화 출력
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# 응답 처리
if user_prompt:
    st.session_state.chat_history.append(("user", user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 시스템 메시지 (명시적 role prompt 삽입)
    full_prompt = (
        "You are a helpful assistant who explains and writes Python code.\n\n"
        f"User: {user_prompt}\nAssistant:"
    )

    with st.chat_message("assistant"):
        with st.spinner("🧠 Mistral 모델이 답변 중입니다..."):
            response = client.text_generation(
                prompt=full_prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95
            )
            st.markdown(response.strip())
            st.session_state.chat_history.append(("assistant", response.strip()))
