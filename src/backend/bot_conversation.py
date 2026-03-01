import os
import logging
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
    
load_dotenv()
logger = logging.getLogger(__name__)

def conversation(current_document_source):

    try:
        # Intialize connection to AstraDB vector store using OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=os.getenv("OPENAI_API")
            )
        
        # Intialize connection to AstraDB vector store using OpenAI embeddings
        vectorstore = AstraDBVectorStore(
            embedding=embeddings,
            api_endpoint=os.getenv("ASTRA_END_POINT"),
            collection_name="document_qa_collection",
            token=os.getenv("ASTRA_TOKEN"),
        )                                                               

        logger.info("Vector store connection established successfully with AstraDB.")
        logger.info("Conversation received source → %s", current_document_source)

        # Create a retriever from the vector store with custom search parameters to filter by document source and limit results
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 3,
            "filter": {
                "source": current_document_source
                }
            })

        logger.info("Retriever created successfully from the vector store.")

        # Initialize the language model (LLM) using OpenAI's GPT-4o-mini with specified temperature and API key
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3, 
            openai_api_key=os.getenv("OPENAI_API")
        )                                                               

        # Create a prompt template for contextualizing the user's question based on the chat history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given chat history and a latest question, rephrase the question to be standalone."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ]
        )                                                                   

        logger.info("Contextualization prompt for history created successfully.")

        # Create a prompt template for question-answering based on retrieved context
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer the question based only on the provided context.\n\nContext:\n{context}"),
                ("human", "{input}")
            ]
        )                                                               

        logger.info("Question-answering prompt created successfully.")

        # Create a history-aware retriever that uses the retriever and contextualization prompt to reformulate user questions based on chat history
        history_aware_retriever = create_history_aware_retriever(
            retriever=retriever, 
            prompt=contextualize_q_prompt, 
            llm=llm
        )                                                               
        
        logger.info("History-aware retriever created successfully.")

        # Create a question-answering chain that uses the question-answering prompt and the language model to generate answers based on retrieved context
        question_answering_chain = create_stuff_documents_chain(
            prompt=qa_prompt,
            llm=llm
        )                                                              
        
        logger.info("Question-answering chain created successfully.")
        
        # Create a retrieval-augmented generation (RAG) chain that combines the history-aware retriever and the question-answering chain to handle user queries in a conversational manner    
        rag_chain = create_retrieval_chain(
            history_aware_retriever, 
            question_answering_chain
        )                                                              
        
        logger.info("RAG chain created successfully.")    
        logger.info("Conversation setup completed successfully. Ready to handle user queries.")
        
        return rag_chain
    
    except Exception as e:
        logger.error("CONVERSATION SETUP ERROR:", exc_info=True)
        raise RuntimeError("An error occurred while setting up the conversation chain. Please check the logs for more details.") from e