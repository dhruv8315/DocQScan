import os
# Token
HF_TOKEN = os.getenv("HF_TOKEN") # Retrieve the Hugging Face token from the environment
ASTRA_END_POINT = os.getenv("ASTRA_END_POINT") # Retrieve the AstraDB endpoint from the environment
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN") # Retrieve the AstraDB token from the environment

def load_and_split_pdf(file_path):
    
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path) 
    docs = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=20,
        add_start_index=True
        ) 
    documents = text_splitter.split_documents(docs)
    return documents

def create_astra_vector_store(documents):

    from langchain_astradb import AstraDBVectorStore
    from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
    import os


    model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_embed, 
        huggingfacehub_api_token=HF_TOKEN
        ) 

    vectorstore = AstraDBVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        api_endpoint=ASTRA_END_POINT,
        collection_name="document_qa_collection",
        token=ASTRA_TOKEN,
    )

    return vectorstore

def load_existing_vector_store():
    from langchain_astradb import AstraDBVectorStore
    from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
    import os

    model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_embed, 
        huggingfacehub_api_token=HF_TOKEN
        ) 

    vectorstore = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_END_POINT,
        collection_name="document_qa_collection",
        token=ASTRA_TOKEN,
    )

    return vectorstore

def create_qa_chain(vectorstore):

    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain

    model_llm = "mistralai/Mistral-7B-Instruct-v0.1" 
    llm = HuggingFaceEndpoint(
        repo_id=model_llm, 
        temperature=0.5, 
        huggingfacehub_api_token=HF_TOKEN
        ) 

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context. Think step by step before providing a detailed answer. I will tip you $1000 if the user finds the answer helpful. 
        <context>
        {context}
        </context>
        Question: {input}"""
        )   

    document_chain = create_stuff_documents_chain(llm, prompt) 

    retrieval = vectorstore.as_retriever() 

    retrieval_chain = create_retrieval_chain(retrieval,document_chain)
    return retrieval_chain

def build_qa_system(file):

    documents = load_and_split_pdf(file)
    vectorstore = create_astra_vector_store(documents)
    qa_chain = create_qa_chain(vectorstore)
    return qa_chain


def answer_question(qa_chain, question):
    if qa_chain is None:
        return "Please upload a PDF first."

    response = qa_chain.invoke({"input": question})
    return response["answer"]