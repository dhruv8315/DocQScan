# Start of import statements

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
import os

# End of import statements

# Tokens and endpoints
ASTRA_END_POINT = os.getenv("ASTRA_END_POINT") # Retrieve the AstraDB endpoint from the environment
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN") # Retrieve the AstraDB token from the environment

def process_pdf(file_path):

    print("Loading PDF document...")
    loader = PyPDFLoader(file_path)              # Load the PDF document using PyPDFLoader
    docs = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        add_start_index=True
        )                                         # Initialize the text splitter with specified chunk size, overlap, and start index
    all_splits = text_splitter.split_documents(docs) 

    print(type(all_splits[0]))
    print(f"Total chunks created: {len(all_splits)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=os.getenv("OPENAI_API")
        )                                           # Initialize the OpenAI embeddings with the specified model and API key
    
    test_vector = embeddings.embed_query("hello world")
    print(len(test_vector))
    
    vector_store = AstraDBVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    api_endpoint=ASTRA_END_POINT,
    collection_name="document_qa_collection",
    token=ASTRA_TOKEN
    )                                                # Create a vector store in AstraDB from the document chunks and their corresponding embeddings
    
    print("Documents embedded and stored in AstraDB successfully.") 