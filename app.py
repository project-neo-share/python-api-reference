import streamlit as st
import google.generativeai as genai

# ✅ 가장 먼저 실행되어야 함
st.set_page_config(page_title="🐍 Gemini 기반 파이썬 코드 챗봇")

# 🔐 사이드바에서 API 키 입력
st.sidebar.title("🔐 Gemini API 키 설정")
api_key_input = st.sidebar.text_input("GOOGLE_API_KEY를 입력하세요", type="password")

if api_key_input:
    try:
        genai.configure(api_key=api_key_input)
        model = genai.GenerativeModel("gemini-pro")
    except Exception as e:
        st.sidebar.error(f"API 설정 실패: {e}")
        st.stop()
else:
    st.warning("API 키를 먼저 입력하세요.")
    st.stop()
    
# 🌐 메인 UI
st.title("💬 Gemini 프로그래밍 조교")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("파이썬 코드 관련 질문을 입력하세요!")

# 🔁 이전 대화 출력
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# 🤖 질문 처리
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    prompt = f"너는 파이썬 전문가야. 다음 질문에 대해 설명과 예제 코드를 제공해줘:\n{user_input}"
    with st.chat_message("assistant"):
        with st.spinner("Gemini가 답변 중입니다..."):
            response = model.generate_content(prompt)
            st.markdown(response.text)
            st.session_state.chat_history.append(("assistant", response.text))
