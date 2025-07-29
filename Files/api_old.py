import pandas as pd
import datetime
import speech_recognition as sr
from typing import List
import os
import subprocess
from langchain.prompts import ChatPromptTemplate
import requests
from langchain.tools import Tool, tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.agents import Tool
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain.vectorstores import Chroma,FAISS,Pinecone,Weaviate
from datetime import datetime
import weaviate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from fastapi import FastAPI,File,Form,UploadFile

def select_vector_store(db_select,docs,embedding):
    if db_select=='FAISS':
        vectordb=FAISS.from_documents(documents=docs,embedding=embedding)
    # elif db_select=='Pinecore':
        # st.text_input("Enter Pinecore API",type='password')
        # vectordb==Pinecone()
    elif db_select=='Weavite':
        client = weaviate.connect_to_local() 
        vectordb=Weaviate.from_documents(docs,embedding=embedding,client=client)
    else:
        vectordb=Chroma.from_documents(documents=docs,embedding=embedding)
    return vectordb

def llm(model,model_option,api_key=None):
    if model=="Open AI":
        llm = ChatOpenAI(model=model_option, api_key=api_key, verbose=1, temperature=0.1)
        embeddings=OpenAIEmbeddings(model=model_option,api_key=api_key)
    elif model == "Ollama":
        llm = ChatOllama(model=model_option, verbose=1, temperature=0.1)
        embeddings=OllamaEmbeddings(model=model_option)
    elif model=="Groq":
        llm = ChatGroq(model=model_option, api_key=api_key, verbose=1, temperature=0.1)
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif model=="Gemini":
        llm = GoogleGenerativeAI(api_key=api_key, verbose=1, temperature=0.1, model=model_option)
        embeddings=GoogleGenerativeAIEmbeddings(model=model_option)
    return embeddings,llm

app=FastAPI()
@app.post("/main")
async def main_app(model:str =Form(),model_name:str =Form(),api_key : str =Form(""),files: List[UploadFile] = File(...), vector_store: str = Form("FAISS"),user_prompt:str =Form()):
    system_prompt="""You are a helpful assistnt that is responsible to provide response by using the provieded doccuments. Explain in detail and provide response in markdown foramat and in around 500 words. provide reference from the doccument and abstrain from answering if no context is provided. be respectful and helpful to the user : 
    the previous context is :{context} 
    user prompt  is :{input}"""
    timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name=f"folder_{timestamp}"
    print(files)
    if files is not None:
        for file in files:
        # Save the uploaded file to the specified folder
            os.makedirs(folder_name,exist_ok=True)
            file_path = os.path.join(folder_name, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
            print(f"File saved successfully to {file_path}")

        print(files)
    prompt_template=ChatPromptTemplate.from_template(system_prompt)
    embeddings,llms=llm(model=model,model_option=model_name,api_key=api_key)
    loader=PyPDFDirectoryLoader(path=folder_name)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=text_splitter.split_documents(documents)
    vectorstore=select_vector_store(db_select=vector_store,docs=docs,embedding=embeddings)
    output_parser=StrOutputParser()
    document_chain=create_stuff_documents_chain(llm=llms,prompt=prompt_template,output_parser=output_parser)
    retriever=vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
    # output_parser=output_parser
    )   
    response=retrieval_chain.invoke({
        # "context":{st.session_state.messages_normal},
        "input":user_prompt})
    response_ans=response['answer']
    return {"output":response_ans}