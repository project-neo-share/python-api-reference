from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import docx
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
from streamlit_chat import message


def main():
    load_dotenv()
    st.set_page_config(page_title="2025-1학기 파이썬프로그래밍(한국공대)")
    st.header("파이썬 API 레퍼런스 챗봇")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 고정된 PDF 파일 경로 리스트
    file_paths = [
        "assets/Programming-Fundamentals-1570222270.pdf",
        "assets/01_교재디자인 내지.pdf",
        "assets/1분파이썬_강의자료_전체.pdf"
    ]
    with st.sidebar:
        process = st.button("파이썬 API 레퍼런스 불러오기")

    if process:
        st.info("📄 텍스트 추출 중입니다...")
        files_text = get_files_text(file_paths)
    
        st.info("📚 텍스트 분할 중입니다...")
        text_chunks = get_text_chunks(files_text)
    
        st.info("🧠 임베딩 및 벡터스토어 생성 중입니다...")
        vetorestore = get_vectorstore(text_chunks)
    
        st.info("💬 챗봇 체인 구성 중입니다...")
        st.session_state.conversation = get_conversation_chain(vetorestore)
    
        st.success("✅ 준비 완료!")
        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("파이썬 API 레퍼런스: 질문해 보세요.")
        if user_question:
            handel_userinput(user_question)

def get_files_text(file_paths):
    text = ""
    for uploaded_file in file_paths:
        _, ext = os.path.splitext(uploaded_file)
        if ext == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif ext == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore):
    handler = StdOutCallbackHandler()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.7,"max_length":256})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory,
        callbacks=[handler]
    )
    return conversation_chain


def handel_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()
