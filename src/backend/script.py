# Start of import statements
import logging
import traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
# End of import statements

def process_pdf(file_path):

    try:
        logger.info("Loading PDF document...")
        
        loader = PyPDFLoader(file_path)              # Load the PDF document using PyPDFLoader
        docs = loader.load() 

        if not docs:
            logger.warning("No content extracted from the PDF.")
            raise ValueError("The uploaded PDF appears to be empty or could not be processed. Please try with a different PDF file.")
        
        # Initialize the text splitter with specified chunk size, overlap, and start index
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            add_start_index=True
            )                                         
        all_splits = text_splitter.split_documents(docs) 

        # Adding custom metadata filtering
        for doc in all_splits:
            doc.metadata["source"] = file_path
        
        logger.info("file source: %s", file_path)
        logger.info("Type of first document: %s", type(all_splits[0]))
        logger.info("Total chunks created: %d", len(all_splits))

        # Initialize the OpenAI embeddings with the specified model and API key
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=os.getenv("OPENAI_API")
            )                                           
        
        # Create a vector store in AstraDB from the document chunks and their corresponding embeddings
        vector_store = AstraDBVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        api_endpoint=os.getenv("ASTRA_END_POINT"),
        collection_name="document_qa_collection",
        token=os.getenv("ASTRA_TOKEN")
        )                                                
        
        logger.info("Documents embedded and stored in AstraDB successfully.")

        return file_path
    
    except Exception as e:
        
        logger.error("PDF PROCESSING ERROR:", exc_info=True)
        raise RuntimeError("An error occurred while processing the PDF. Please check the logs for more details.") from e