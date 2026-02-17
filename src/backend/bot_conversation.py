import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()
def conversation():

    model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_embed, 
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
        ) 

    vectorestore = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=os.getenv("ASTRA_END_POINT"),
        collection_name="document_qa_collection",
        token=os.getenv("ASTRA_TOKEN"),
    )

    retriever = vectorestore.as_retriever(search_kwargs={"k": 3})

    model_llm = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=model_llm, 
        temperature=0, 
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given chat history and a latest question, rephrase the question to be standalone."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based only on the provided context."),
            MessagesPlaceholder(variable_name="context"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        retriever=retriever, 
        prompt=contextualize_q_prompt, 
        llm=llm
        )
    
    question_answering_chain = create_stuff_documents_chain(
        prompt=qa_prompt,
        llm=llm
        )
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever, 
        question_answering_chain
        )
    return rag_chain