import streamlit as st
import pandas as pd
import datetime
import speech_recognition as sr
import pyttsx3
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
def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except subprocess.CalledProcessError as e:
        print("Error running 'ollama list':", e)
        print("Output:", e.stdout)
        print("Error Output:", e.stderr)

def get_all_groq_models(api_key: str) -> list:
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
    except requests.RequestException as e:
        print(f"Error fetching Groq models: {e}")
        return []

def get_all_gemini_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
        return gemini_models
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def llm():
    # with st.sidebar:
    st.title("Model Selection")
    model_option = None
    openai_api_key = " "
    model_list = ["Ollama", "Open AI", "Groq", "Gemini"]
    model = st.selectbox("Select Model", model_list)

    if model == "Open AI":
        openai_api_key = st.text_input("Enter your OpenAi API key", type="password")
        st.divider()
        if openai_api_key:
            model_option = st.selectbox(
                "Select AI Model",
                ["gpt-3.5-turbo (Fast)", "gpt-4o (High Quality)"],
                help="GPT 3.5 is faster than GPT 4"
            )
            llm = ChatOpenAI(model=model_option, api_key=openai_api_key, verbose=1, temperature=0.1)
            embeddings=OpenAIEmbeddings(model=model_option,api_key=openai_api_key)
        else:
            st.warning("enter API Key")

    elif model == "Ollama":
        ollama_list = get_ollama_models()
        model_option = st.selectbox("Select AI Model", ollama_list)
        llm = ChatOllama(model=model_option, verbose=1, temperature=0.1)
        embeddings=OllamaEmbeddings(model=model_option)

    elif model == "Groq":
        groqapi_key = st.text_input("Enter your Groq API key", type="password")
        st.divider()
        openai_api_key = groqapi_key
        if groqapi_key:
            groq_list = get_all_groq_models(api_key=groqapi_key)
            model_option = st.selectbox("Select AI Model", groq_list)
            llm = ChatGroq(model=model_option, api_key=groqapi_key, verbose=1, temperature=0.1)
            embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            st.warning("enter API Key")

    elif model == "Gemini":
        Geminiapi_key = st.text_input("Enter your Gemini API key", type="password")
        st.divider()
        openai_api_key = Geminiapi_key
        if Geminiapi_key:
            gemini_list = get_all_gemini_models(api_key=Geminiapi_key)
            model_option = st.selectbox("Select AI Model", gemini_list)
            llm = GoogleGenerativeAI(api_key=Geminiapi_key, verbose=1, temperature=0.1, model=model_option)
            embeddings=GoogleGenerativeAIEmbeddings(model=model_option)
        else:
            st.warning("enter API Key")

    return llm,embeddings
def select_vector_db(docs,embedding):
    list_vd=['FAISS', 
            #  'Pinecone',
             'Weaviate', 'ChromaDB']
    db_select=st.selectbox("Select the Vector DB Provider",list_vd,index=0)
    if db_select=='FAISS':
        vectordb=FAISS.from_documents(documents=docs,embedding=embedding)
    elif db_select=='Pinecore':
        st.text_input("Enter Pinecore API",type='password')
        vectordb==Pinecone()
    elif db_select=='Weavite':
        client = weaviate.connect_to_local() 
        vectordb=Weaviate.from_documents(docs,embedding=embedding,client=client)
    else:
        vectordb=Chroma.from_documents(documents=docs,embedding=embedding)
    return vectordb
st.title("LLM with doccument support")
st.text("Upload Doccuments and using then ask different questions to the llm")
# no_of_docs=st.slider("Enter no of doccuments",1,20)
timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# if st.session_state.folder_name is None:
st.session_state.folder_name=f"folder_{timestamp}"
files=st.file_uploader("Enter pdf files",'pdf',True)
if files is not None:
    
    for file in files:
    # Save the uploaded file to the specified folder
        os.makedirs(st.session_state.folder_name,exist_ok=True)
        file_path = os.path.join(st.session_state.folder_name, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
        st.success(f"File saved successfully to {file_path}")

print(files)
# st.write(files)
st.session_state.messages_normal=[]
st.session_state.messages=[]
st.session_state.system_prompt="""You are a helpful assistnt that is responsible to provide response by using the provieded doccuments. Explain in detail and provide response in markdown foramat and in around 500 words. provide reference from the doccument and abstrain from answering if no context is provided. be respectful and helpful to the user : 
the previous context is :{context} 
user prompt  is :{input}"""

prompt_template=ChatPromptTemplate.from_template(st.session_state.system_prompt)
system_prompt=SystemMessage(content=st.session_state.system_prompt)
st.session_state.messages.append(system_prompt)
st.session_state.messages_normal.append(("system",system_prompt))
st.session_state.llms,st.session_state.embedding=llm()
st.session_state.loader=PyPDFDirectoryLoader(path=st.session_state.folder_name)
st.session_state.documents=st.session_state.loader.load()
st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
st.session_state.docs=st.session_state.text_splitter.split_documents(st.session_state.documents)
st.session_state.vectorstore=select_vector_db(docs=st.session_state.docs,embedding=st.session_state.embedding)
# similarity_search=st.text_input("Enter Text for similarity search")
# if similarity_search:
#     st.write(st.session_state.vectorstore.similarity_search(similarity_search))
# print(st.session_state)
output_parser=StrOutputParser()
# st.write(st.session_state)
user_prompt=st.text_input("enter your querry to search from ")
user_prompt_hum=HumanMessage(content=user_prompt)
st.session_state.messages.append(user_prompt_hum)
st.session_state.messages_normal.append(("user",user_prompt))
document_chain=create_stuff_documents_chain(llm=st.session_state.llms,prompt=prompt_template,output_parser=output_parser)
retriever=st.session_state.vectorstore.as_retriever()
                                            # prompt=PromptTemplate(template=st.session_state.system_prompt,input_variables=["user_prompt"])
# retrieval_chain=create_retrieval_chain(retriever,document_chain,output_parser=output_parser)
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
    # output_parser=output_parser
)
# response=retrieval_chain.invoke(st.session_state.messages)
if st.button("submit"):
    response=retrieval_chain.invoke({
        # "context":{st.session_state.messages_normal},
        "input":user_prompt})
    st.session_state.messages.append(response)
    response_ans=response['answer']
    st.session_state.messages_normal.append(response_ans)
    st.markdown(response_ans)
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
# print(llm)