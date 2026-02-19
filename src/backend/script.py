# Start of import statements

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
import os

# End of import statements

# Token
HF_TOKEN = os.getenv("HF_TOKEN") # Retrieve the Hugging Face token from the environment
ASTRA_END_POINT = os.getenv("ASTRA_END_POINT") # Retrieve the AstraDB endpoint from the environment
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN") # Retrieve the AstraDB token from the environment

def process_pdf(file_path):

    print("Loading PDF document...")

    loader = PyPDFLoader(file_path) 
    
    docs = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        add_start_index=True
        ) 

    all_splits = text_splitter.split_documents(docs) 

    
    
    print(type(all_splits[0]))
    print(f"Total chunks created: {len(all_splits)}")

    
    
    """model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEndpointEmbeddings(model=model_embed, huggingfacehub_api_token=HF_TOKEN) """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=os.getenv("OPENAI_API")
        )
    test_vector = embeddings.embed_query("hello world")
    print(len(test_vector))
    
    vector_store = AstraDBVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    api_endpoint=ASTRA_END_POINT,
    collection_name="document_qa_collection",
    token=ASTRA_TOKEN
    )
    
    print("Documents embedded and stored in AstraDB successfully.") 
   
    """model_llm = "mistralai/Mistral-7B-Instruct-v0.3" 
    
    llm = HuggingFaceEndpoint( 
        repo_id=model_llm, 
        temperature=0.5, 
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
        ) 

    """    
    #prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context. Think step by step before providing a detailed answer. You are a helpful assistant that only uses provided context. 
        <context>
        {context}
        </context>
        Question: {input}"""
    #)   
    """
 document_chain = create_stuff_documents_chain(llm, prompt) 

    retrieval = vector_store.as_retriever() 

    retrieval_chain = create_retrieval_chain(retrieval,document_chain)

    results = retrieval_chain.invoke({"input": "Name of candidate?"}) 
    
    return results["answer"]"""