# Start of import statements

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_astradb import AstraDBVectorStore
import getpass
import os

# End of import statements

# Token
HF_TOKEN = os.getenv("HF_TOKEN") # Retrieve the Hugging Face token from the environment

def process_pdf(file_path, text_input=""):

    loader = PyPDFLoader(file_path) # Create an instance of the PyPDFLoader class, passing the file path of the uploaded PDF as an argument. This loader is responsible for loading and processing the PDF document.
    
    docs = loader.load() #  This method reads the PDF file and extracts its content, returning it as a list of documents (docs).

    #Debug : print(len(docs)) # Print the extracted documents to the console for debugging purposes.
    #print(f"{docs[0].page_content[:200]}\n")
    #print(docs[0].metadata)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,add_start_index=True) # Create an instance of the RecursiveCharacterTextSplitter class, which is used to split the extracted documents into smaller chunks. The chunk_size parameter specifies the maximum size of each chunk, while the chunk_overlap parameter defines the number of characters that overlap between consecutive chunks. The add_start_index parameter indicates whether to include the starting index of each chunk in the output. And it should not increase the token size.

    all_splits = text_splitter.split_documents(docs) # This method takes the list of extracted documents and splits them into smaller chunks based on the specified chunk size and overlap. The resulting chunks are stored in the all_splits variable.

    #print(all_splits)
    #print(f"Total chunks created: {len(all_splits)}") # Print the total number of chunks created to the console for debugging purposes.

    
    model_llm = "facebook/rag-token-base" # Define the model ID for the Hugging Face model that will be used for embedding the document chunks. This model is designed for document question answering tasks.
    
    llm = HuggingFaceEndpoint(repo_id=model_llm, temperature=0.5, huggingfacehub_api_token=HF_TOKEN) # Create an instance of the HuggingFaceEndpoint class, which allows you to interact with the specified Hugging Face model. The repo_id parameter specifies the model repository, the task parameter indicates that the model will be used for feature extraction, and the huggingfacehub_api_token parameter provides the necessary authentication token.
    print("Completed endpoint setup...") # Print a message to the console indicating that the endpoint setup is complete.

    model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEndpointEmbeddings(model=model_embed, huggingfacehub_api_token=HF_TOKEN) # Create an instance of the HuggingFaceEndpointEmbeddings class, which is responsible for generating embeddings for the document chunks using the specified Hugging Face model.

    query_result = embeddings.embed_documents(all_splits) # Generate an embedding for the input text query using the embeddings instance. This will be used later for similarity search in the vector store. 

  
    #print("Connecting to AstraDB and creating vector store...") # Print a message to the console indicating that the connection to AstraDB is being established and the vector store is being created.

    vector_store = AstraDBVectorStore(
    embedding=query_result,
    api_endpoint="https://4b9d8e60-d188-4473-91eb-2090c74d8f62-us-east1.apps.astra.datastax.com",
    collection_name="document_qa",
    token="AstraCS:jnnfqbitrpcIJAkYiQDKrEXo:92a2215a9f986619d90db4dda89c4e3c4f5f6bd8aecedf4a2a2043f5548b7d05",
    namespace="default_keyspace",
    )
    
    #print("Adding documents to vector store...") # Print a message to the console indicating that the documents are being added to the vector store.
    ids = vector_store.add_documents(documents=all_splits)
    
    results = vector_store.similarity_search(text_input)
    """
    #print(f"Similarity search results: {results[0].page_content}") # Print the results of the similarity search to the console for debugging purposes."""
    return results