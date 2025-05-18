import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="🐍 Gemini 기반 파이썬 코드 챗봇")

st.sidebar.title("🔐 Gemini API 키 입력")
api_key = st.sidebar.text_input("GOOGLE_API_KEY", type="password")

if not api_key:
    st.warning("API 키를 입력해주세요.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-pro")  # ✅ 수정된 부분
except Exception as e:
    st.error(f"API 초기화 실패: {e}")
    st.stop()

st.title("💬 Gemini 파이썬 코드 질문 챗봇")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("파이썬 코드 관련 질문을 입력하세요!")

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Gemini가 답변 중입니다..."):
            prompt = f"너는 파이썬 전문가야. 다음 질문에 대해 설명과 예제 코드를 제공해줘:\n{user_input}"
            response = model.generate_content(prompt)
            st.markdown(response.text)
            st.session_state.chat_history.append(("assistant", response.text))
