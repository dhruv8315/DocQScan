from langchain_community.vectorstores.cassandra import Cassandra
from langchain_community.vectorstores import VectorStoreIndexWrapper
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from  langchain_text_splitters import CharacterTextSplitter
# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
import cassio
from PyPDF2 import PdfReader
from typing_extensions import Concatenate



def process_pdf(file_obj):
     
    file_path = file_obj.name  #Access the file path using file_obj
    reader = PdfReader(file_path) # Use PdfReader to read the PDF file and store the content in a reader

    raw_text = '' # Initialize an empty string to store the extracted text from the PDF

    for i, pages in enumerate(reader.pages): # Loop through each page in the PDF and extract the text content
        content = pages.extract_text()
        if content: # Check if the content is not empty before appending it to the raw_text variable
            raw_text += content
    
    text_splitter = CharacterTextSplitter(seperator =  "\n", chunk_size=800, chunk_overlap=200) # Initialize a CharacterTextSplitter with a chunk size of 1000 characters and an overlap of 200 characters between chunks

    texts = text_splitter.split_text(raw_text) # Use the text splitter to split the raw text into smaller chunks and store them in a list called texts