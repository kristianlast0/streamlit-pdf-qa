import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmltemplates import css, bot_template, user_template
from langchain.llms import CTransformers, HuggingFaceHub

def get_text(files) -> str:
    raw_text = []
    for file in files:
        if file.type == 'application/pdf':
            for page in PdfReader(file).pages:
                raw_text.append(page.extract_text())
        elif file.type == 'text/plain':
            raw_text.append(file.read())
    return " ".join(raw_text)

def get_text_chunks(text: str):
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks: list):
    # embeddings = HuggingFaceInstructEmbeddings(model_name='distilbert-base-uncased')
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = CTransformers(
    #     model="llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type='llama',
    #     max_new_tokens=2000,
    #     temperature=0.5,
    # )
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={'temperature': 0.2, 'max_length': 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(question):
    response = st.session_state.conversation({'question': question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title='Hello World', page_icon='🧊')
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state: st.session_state.conversation = None
    st.header('Hello World')
    user_question = st.text_input('Ask a question about your documents')
    if user_question: handle_userinput(user_question)
    st.write(bot_template.format('Hello World'), unsafe_allow_html=True)
    st.write(user_template.format('Hello World'), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader('Documents')
        files = st.file_uploader('Upload documents', type=['pdf', 'txt'], accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner("Processing..."):
                raw_text = get_text(files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()