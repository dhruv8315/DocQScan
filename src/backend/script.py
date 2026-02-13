from langchain_community.vectorstores.cassandra import Cassandra
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_astradb import AstraDBVectorStore
#from langchain.indexes.vectorstore import VectorStoreIndexCreator   
#from langchain_community.vectorstores import VectorStoreIndexWrapper
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from  langchain_text_splitters import CharacterTextSplitter
# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
import cassio
from PyPDF2 import PdfReader
from openai import embeddings
from typing_extensions import Concatenate

ASTRA_DB_APPLICATION_TOKEN= "AstraCS:ZIBtulKPjMpbUarDaKlhhOEH:bcf17aee02f6925a715549b271241060cfb3db64195d0635118e2f1fc5d65588"

ASTRA_DB_ID = "4b9d8e60-d188-4473-91eb-2090c74d8f62"

ASTRA_DB_API_ENDPOINT = "https://4b9d8e60-d188-4473-91eb-2090c74d8f62-us-east1.apps.astra.datastax.com"

OPEN_API_KEY = "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"


def process_pdf(file_obj,query_text=""):
     
    file_path = file_obj.name  #Access the file path using file_obj
    reader = PdfReader(file_path) # Use PdfReader to read the PDF file and store the content in a reader

    raw_text = '' # Initialize an empty string to store the extracted text from the PDF

    for i, pages in enumerate(reader.pages): # Loop through each page in the PDF and extract the text content
        content = pages.extract_text()
        if content: # Check if the content is not empty before appending it to the raw_text variable
            raw_text += content
    
    text_splitter = CharacterTextSplitter(separator =  "\n", chunk_size=800, chunk_overlap=200) # Initialize a CharacterTextSplitter with a chunk size of 1000 characters and an overlap of 200 characters between chunks

    texts = text_splitter.split_text(raw_text) # Use the text splitter to split the raw text into smaller chunks and store them in a list called texts

    """Create an instance of the OpenAIEmbeddings class, which will be used to generate embeddings for the text chunks. And create llm instance of the OpenAI class, which will be used to generate responses from the language model."""
    llm = OpenAI(openai_api_key=OPEN_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

    """
        cassio.init(
            application_token=ASTRA_DB_APPLICATION_TOKEN,
            database_id=ASTRA_DB_ID,
        )"""
    """Create an instance of the Cassandra vector store, which will be used to store the embeddings of the text chunks. The session and keyspace parameters are set to None, as they will be automatically created by CassIO when the vector store is initialized."""
    astra_vector_store = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="docement_qa",
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace="default_keyspace",
    )

    vector_store_index = astra_vector_store.add_documents(documents=texts) # Add the top 50 text chunks to the Cassandra vector store, which will generate embeddings for each chunk and store them in the database

   
    #if query_text != "": # Check if a query is provided
    answer = vector_store_index.similarity_search(query=query_text) # Use the query method of the Cassandra vector store to perform a similarity search for the query text and generate a response using the language model. The response is stored in the answer variable