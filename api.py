import os
from typing import List,Optional
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS, Weaviate
import weaviate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
# import pyodbc


def select_vector_store(db_select: str, docs, embedding):
    db_select=db_select.strip()
    db_select=db_select.lower()
    if db_select == 'faiss':
        vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
    elif db_select == 'weaviate':
        client = weaviate.connect_to_local()
        vectordb = Weaviate.from_documents(documents=docs, embedding=embedding, client=client)
    else:
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding)
    return vectordb


def llm(model: str, model_option: str, api_key: str = None):
    if model == "Open AI":
        llm = ChatOpenAI(model=model_option, api_key=api_key, verbose=True, temperature=0.1)
        embeddings = OpenAIEmbeddings(model=model_option, api_key=api_key)
    elif model == "Ollama":
        llm = ChatOllama(model=model_option, verbose=True, temperature=0.1)
        embeddings = OllamaEmbeddings(model=model_option)
    elif model == "Groq":
        llm = ChatGroq(model=model_option, api_key=api_key, verbose=True, temperature=0.1)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif model == "Gemini":
        llm = GoogleGenerativeAI(api_key=api_key, verbose=True, temperature=0.1, model=model_option)
        embeddings = GoogleGenerativeAIEmbeddings(model=model_option)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return embeddings, llm


app = FastAPI()




@app.post("/main")
async def main_app(
    model: str = Form(...),
    model_name: str = Form(...),
    api_key: str = Form(""),
    files: List[UploadFile] = File(...),
    vector_store: str = Form("FAISS"),
    user_prompt: str = Form(...)
):
    
    system_prompt = (
        "You are a helpful assistant that is responsible to provide a response "
        "using the provided documents. Explain in detail and provide response "
        "in markdown format and in around 500 words. Provide references from the document "
        "and abstain from answering if no context is provided. Be respectful and helpful to the user. "
        "The previous context is: {context} \nUser prompt is: {input}"
    )

    # !cratin foldr with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"folder_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    for file in files:
        file_path = os.path.join(folder_name, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

    prompt_template = ChatPromptTemplate.from_template(system_prompt)

    embeddings, llms = llm(model=model, model_option=model_name, api_key=api_key)

    loader = PyPDFDirectoryLoader(path=folder_name)
    documents = loader.load()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found in uploaded files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    if not docs:
        raise HTTPException(status_code=400, detail="Failed to split documents into chunks.")

    vectorstore = select_vector_store(db_select=vector_store, docs=docs, embedding=embeddings)

    output_parser = StrOutputParser()

    document_chain = create_stuff_documents_chain(
        llm=llms,
        prompt=prompt_template,
        output_parser=output_parser
    )

    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": user_prompt})
    print(response)

    response_ans = response.get('answer', '')

    # ! similarity search
    context_docs = [doc.page_content for doc in response.get('context', [])]

    return {"output": response_ans,"context_docs":context_docs}


@app.post('/upload_docs')
async def upload_docs(files: List[UploadFile] = File(...)):
    try:# !cratin folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"folder_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_name, file.filename)
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
        return {"result":"Successfully uploaded doccuments"}
    except Exception as e:
        return {"result":f"Return error while uploading doccuments : {e}"}
    
@app.post('/docs_metadata')
async def docs_metadata(
    files: List[UploadFile] = File(...),
    vector_store: str = Form("FAISS"),
    user_prompt: Optional[str] = Form(None)
 ):
    system_prompt = (
        "You are a helpful assistant that is responsible to provide a response "
        "using the provided documents. Explain in detail and provide response "
        "in markdown format and in around 500 words. Provide references from the document "
        "and abstain from answering if no context is provided. Be respectful and helpful to the user. "
        "The previous context is: {context} \nUser prompt is: {input}"
    )

    # !cratin foldr with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"folder_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    files_name=[]
    files_size=[]
    files_path=[]

    for file in files:
        file_path = os.path.join(folder_name, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        files_name.append(file.filename)
        files_path.append(file_path)
        files_size.append(len(contents))
    print("done 1")
    # embeddings, llms = llm(model=model, model_option=model_name, api_key=api_key)

    loader = PyPDFDirectoryLoader(path=folder_name)
    documents = loader.load()
    print("done 2")

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found in uploaded files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print("done 3")

    if not docs:
        raise HTTPException(status_code=400, detail="Failed to split documents into chunks.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("done 4")

    vectorstore = select_vector_store(db_select=vector_store, docs=docs, embedding=embeddings)
    simi_result=""
    print("done 5")

    if user_prompt:
        simi_result=vectorstore.similarity_search(user_prompt)
        print(simi_result)
    print("done 6")
    

    return {
        "files_name":files_name,
        "files_path":files_path,
        "files_size":files_size,
        "similarity_search":simi_result if simi_result else "",
        "chunks":docs,
        "no_of_chunks":len(docs),
    }