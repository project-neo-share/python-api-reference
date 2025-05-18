from dotenv import load_dotenv
import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="💬 파이썬 코드 도우미 챗봇")
    st.header("👨‍💻 파이썬 프로그래밍 질문 챗봇 (무료 Hugging Face 기반)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.chat_input("파이썬 코드에 대해 무엇이든 물어보세요!")
    if user_question:
        handle_user_input(user_question)

def get_conversation_chain():
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Chat 구조 지원
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
        callbacks=[StdOutCallbackHandler()]
    )
    return conversation_chain

def handle_user_input(question):
    response = st.session_state.conversation.run(question)
    st.session_state.chat_history = st.session_state.conversation.memory.chat_memory.messages

    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

if __name__ == "__main__":
    main()
