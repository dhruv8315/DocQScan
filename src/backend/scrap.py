import os
# Token
HF_TOKEN = os.getenv("HF_TOKEN") # Retrieve the Hugging Face token from the environment
ASTRA_END_POINT = os.getenv("ASTRA_END_POINT") # Retrieve the AstraDB endpoint from the environment
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN") # Retrieve the AstraDB token from the environment

def load_split_and_create_vectorstore(file_path):
    
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