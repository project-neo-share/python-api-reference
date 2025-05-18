from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
import os


def main():
    load_dotenv()
    st.set_page_config(page_title="📘 파이썬 API 레퍼런스 챗봇")
    st.header("2025-1학기 파이썬프로그래밍 (한국공대)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    # 고정된 PDF 파일만 사용
    file_paths = [
        "assets/Programming-Fundamentals-1570222270.pdf",
        "assets/1분파이썬_강의자료_전체.pdf"
    ]

    st.info("📄 PDF 텍스트 추출 중...")
    files_text = get_pdf_files_text(file_paths)

    st.info("✂️ 텍스트 분할 중...")
    text_chunks = get_text_chunks(files_text)

    st.info("🔎 벡터 임베딩 중...")
    vectorstore = get_vectorstore(text_chunks)

    st.info("🤖 챗봇 체인 구성 중...")
    st.session_state.conversation = get_conversation_chain(vectorstore)

    st.success("✅ 준비 완료! 질문을 입력하세요.")
    st.session_state.processComplete = True

    # 사용자 질문 입력창
    if st.session_state.processComplete:
        user_question = st.chat_input("파이썬 API 레퍼런스에 대해 무엇이든 물어보세요.")
        if user_question:
            handle_user_input(user_question)


def get_pdf_files_text(file_paths):
    """여러 PDF에서 텍스트 추출"""
    text = ""
    for pdf in file_paths:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(chunks):
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.7, "max_length": 256}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        callbacks=[StdOutCallbackHandler()]
    )


def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)


if __name__ == "__main__":
    main()
