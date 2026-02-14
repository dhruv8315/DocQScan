# Start of import statements

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import AstraDB

import os

# End of import statements

# Token
HF_TOKEN = os.getenv("HF_TOKEN") # Retrieve the Hugging Face token from the environment

def process_pdf(file_path, text_input=""):

    loader = PyPDFLoader(file_path) 
    
    docs = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20,add_start_index=True) 

    all_splits = text_splitter.split_documents(docs) 
    
   
   
   


    model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEndpointEmbeddings(model=model_embed, huggingfacehub_api_token=HF_TOKEN) 
    
    query_result = embeddings.embed_documents([doc.page_content for doc in all_splits]) 
    
    
    vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint="https://4b9d8e60-d188-4473-91eb-2090c74d8f62-us-east1.apps.astra.datastax.com",
    collection_name="document_qa",
    token="AstraCS:jnnfqbitrpcIJAkYiQDKrEXo:92a2215a9f986619d90db4dda89c4e3c4f5f6bd8aecedf4a2a2043f5548b7d05",
    namespace="default_keyspace",
    )
    
    vector_store.add_documents(documents=all_splits)

    print("Documents added to AstraDB vector store successfully.") 

    """
    db = AstraDB.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name="document_qa"
    )"""
    
    
    
    #results = vector_store.similarity_search(text_input)
    """
    #print(f"Similarity search results: {results[0].page_content}") # Print the results of the similarity search to the console for debugging purposes.
    """
    

    
   
   
   
   
   
    model_llm = "facebook/rag-token-base" 
    
    llm = HuggingFaceEndpoint(repo_id=model_llm, temperature=0.5, huggingfacehub_api_token=HF_TOKEN) 

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context. Think step by step before providing a detailed answer. I will tip you $1000 if the user finds the answer helpful. 
        <context>
        {context}
        </context>
        Question: {input}"""
        )   

    document_chain = create_stuff_documents_chain(llm, prompt) 

    retrieval = vector_store.as_retriever() 

    retrieval_chain = create_retrieval_chain(retrieval,document_chain)

    results = retrieval_chain.invoke({"input": text_input}) 
    
    return results["answer"]