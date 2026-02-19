import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()
def conversation():

    """model_embed = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_embed, 
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
        ) """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=os.getenv("OPENAI_API")
        )
    
    vectorstore = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=os.getenv("ASTRA_END_POINT"),
        collection_name="document_qa_collection",
        token=os.getenv("ASTRA_TOKEN"),
    )

    print("Vector store connection established successfully with AstraDB.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Retriever created successfully from the vector store.")

    """ model_llm = "HuggingFaceH4/zephyr-7b-beta"
    llm = HuggingFaceEndpoint(
        repo_id=model_llm,
        task="conversational", 
        temperature=0, 
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.3, 
        openai_api_key=os.getenv("OPENAI_API")
        )

    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given chat history and a latest question, rephrase the question to be standalone."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    print("Contextualization prompt for history created successfully.")

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based only on the provided context.\n\nContext:\n{context}"),
            ("human", "{input}")
        ]
    )

    print("Question-answering prompt created successfully.")

    history_aware_retriever = create_history_aware_retriever(
        retriever=retriever, 
        prompt=contextualize_q_prompt, 
        llm=llm
        )
    
    print("History-aware retriever created successfully.")

    question_answering_chain = create_stuff_documents_chain(
        prompt=qa_prompt,
        llm=llm
        )
    
    print("Question-answering chain created successfully.")

    rag_chain = create_retrieval_chain(
        history_aware_retriever, 
        question_answering_chain
        )
    
    print("RAG chain created successfully.")

    """def convert_gradio_to_langchain(history):
        lc_history = []
        for msg in history:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))
        return lc_history
    
    
    def chat(user_message, history):
        lc_history = convert_gradio_to_langchain(history)
        
        response = rag_chain.invoke({
            "input": user_message,
            "chat_history": lc_history
            })
        
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response["answer"]})

        return response["answer"]"""
    
    print("Conversation setup completed successfully. Ready to handle user queries.")
    
    return rag_chain