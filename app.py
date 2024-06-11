import os
import streamlit as st
from langchain_groq import  ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vector store db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # vector embeddings

from dotenv import load_dotenv
load_dotenv()

##load GROQ and Google API from .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")
llm = ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the provided context only.
    Please provide the most accurate response vased on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ##Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  ##document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ##text splitter 
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) ##split documents  
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) ##vector store   
    

prompt1 = st.text_input("What you want from the documents")

if st.button("Creating vector store"):
    vector_embedding()
    st.write("Vector store db is ready")

import time
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriver = st.session_state.vectorstore.as_retriever()
    retriever_chain = create_retrieval_chain(retriver,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---")
     


