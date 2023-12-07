import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from htmltemplates import css, bot_template, user_template
from langchain.llms import CTransformers, HuggingFaceHub
import os, hashlib

def extract_text(file) -> str:
    """Get the text from a pdf or txt file"""
    # if the file is a pdf, extract the text from each page
    if file.type == 'application/pdf':
        # extract the text from each page
        return " ".join([page.extract_text() for page in PdfReader(file).pages])
    # if the file is a txt, read the text
    elif file.type == 'text/plain':
        # read the text from the file
        return file.getvalue().decode('utf-8')
    # otherwise, return None
    return None

def chunker(text: str, size: int = 512, overlap: int = 64):
    """Split text into chunks of size `size` with overlap `overlap`"""
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

def get_vectorstore(file) -> FAISS:
    # load OpenAI embeddings - add OPENAI_API_KEY to .env
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # load the huggingface embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    # get a checksum of the file
    checksum = hashlib.sha256(file.getvalue()).hexdigest()
    # if the vectorstore already exists, load it
    if os.path.exists(f'./vectorstore/{checksum}'):
        # return the vectorstore
        return FAISS.load_local(f'./vectorstore/{checksum}', embeddings=embeddings)
    # otherwise, create it
    text = extract_text(file)
    # split the text into chunks
    chunks = chunker(text, size=384)
    # create the vectorstore
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    # save the vectorstore to disk
    vectorstore.save_local(f'./vectorstore/{checksum}')
    # return the vectorstore
    return vectorstore

def query_vectorstore(vectorstore, query: str, k: int = 1) -> list:
    # query the vectorstore for similar documents and return the top 1
    return vectorstore.similarity_search(query=query, k=k)

def get_qa_chain():
    # use chatGPT model - add OPENAI_API_KEY to .env
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # use a downloaded model (e.g ./models/llama-2-7b-...)
    model = os.path.join(os.getcwd(), 'models', 'llama-2-7b-chat.ggmlv3.q8_0.bin')
    # load the llm
    llm = CTransformers(
        model=model, # local model
        model_type='llama',
        max_new_tokens=128,
        temperature=0.3,
    )
    # load the qa chain
    return load_qa_chain(llm, chain_type="stuff")

def run(query: str) -> None:
    # write the query to the page
    st.write(user_template.format(query), unsafe_allow_html=True)
    # query the vectorstore for similar documents matching the query
    docs = query_vectorstore(st.session_state.vectorstore, query, k=10)
    # write the number of documents found to the page
    st.write(f"Found {len(docs)} documents matching your query")
    # get the answer from the QA chain
    answer = st.session_state.conversation.run(input_documents=docs, question=query)
    # Write the response to the page
    st.write(bot_template.format(answer), unsafe_allow_html=True)

def main():
    # init configurations
    load_dotenv()
    # set up page config
    st.set_page_config(page_title='Document QA', page_icon='🧊')
    # set up page layout
    st.write(css, unsafe_allow_html=True)
    # set up page content
    st.header('Query your Documents')
    # set up session state if not already set up
    if "conversation" not in st.session_state:
        # get the qa chain and put it in session state
        st.session_state.conversation = get_qa_chain()
    # create the sidebar
    with st.sidebar:
        # create a header
        st.subheader('Document Uploads')
        # create a file uploader
        file = st.file_uploader('Upload a document (PDF or TXT)', type=['pdf', 'txt'])
        # otherwise, process the file
        if st.button('Process'):
            # show a spinner while processing
            with st.spinner("Processing file..."):
                # get the vectorstore and put it in session state
                st.session_state.vectorstore = get_vectorstore(file)
    # if no file is uploaded
    if file is None:
        # show a message to upload a file
        st.info('Upload a document to get started')
    else:
        # input for user query capture
        query = st.text_input('Ask a question about your documents')
        # handle user input if query is not empty
        if query: run(query)

if __name__ == '__main__':
    main()