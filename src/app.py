import traceback
import logging
from backend.logger_config import setup_logger
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from backend.bot_conversation import conversation
from backend.script import process_pdf

logger = setup_logger()
    
# Global Variables
qa = None
document_ready = False
current_document_source = None

#upload_pdf function is responsible for handling the PDF file uploaded by the user, processing it, and preparing it for question-answering.
def upload_pdf(file):
    global qa,document_ready,current_document_source

    try:
        logger.info("Processing uploaded PDF...")
        
        if file is None:
            logger.warning("No file uploaded.")
            return gr.Warning("Please upload a PDF file to proceed.")
        
        current_document_source = process_pdf(file)
        logger.info("DEBUG: Stored source in app.py → %s", current_document_source)
        
        qa = None
        document_ready = True

        logger.info("Document ingestion completed successfully.")
        
    except Exception as e:
        logger.error("PDF UPLOAD ERROR:", exc_info=True)

        document_ready = False
        qa = None
        current_document_source = None

        return gr.Warning("Failed to process the PDF. Please try again with a another PDF file.")

#add_text function is responsible for adding the user's input text to the chat history and preparing it for processing by the bot.
def add_text(history, text):
    if history is None:
        history = []
    history = history + [{"role": "user", "content": text}]
    return history, ""

#convert_gradio_history function converts the chat history from Gradio's format to a format compatible with LangChain's message structure, allowing the conversation chain to process the history effectively.
def convert_gradio_history(history):
    lc_history = []
    for msg in history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))
    return lc_history

#bot function is responsible for handling the user's queries, invoking the conversation chain to generate responses based on the processed PDF document, and updating the chat history accordingly.
def bot(history):
    global qa,document_ready,current_document_source

    try:
        if not document_ready:
            logger.warning("User asked question before uploading document.")
            history.append(
                {
                    "role": "assistant", 
                    "content": "Please upload a PDF before asking questions."
                }
            )
            return history

        #Lazy initialization of the conversation chain to ensure it's created only after the document is processed
        if qa is None:
            logger.info("Initializing the conversation chain...")
            logger.info("Passing source to conversation → %s", current_document_source)
            qa = conversation(current_document_source)

        raw_content = history[-1]["content"]
        if isinstance(raw_content, list):
            user_msg = raw_content[0]["text"]
        else:
            user_msg = raw_content

        logger.info(f"User query → %s", user_msg)

        lc_history = convert_gradio_history(history[:-1])

        response = qa.invoke({
            "input": user_msg,
            "chat_history": lc_history
        })
        history.append(
            {
                "role": "assistant", 
                "content": response["answer"]
            }
        )
        return history
    
    except Exception as e:
        logger.error("BOT ERROR:", exc_info=True)

        history.append(
            {
                "role": "assistant", 
                "content": "⚠️ Something went wrong while processing your question."
            }
        )
        return history


# Gradio UI Setup
with gr.Blocks() as demo:

    with gr.Row(scale=3):
        file = gr.File(label="Upload PDF",file_types=[".pdf"],container=False)
        file.change(upload_pdf, file)

    chatbot = gr.Chatbot([], elem_id="chatbot",show_label=False,height=550)

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                show_label=False, 
                placeholder="Ask Anything !",
                container=False
                )
        

        with gr.Column(scale=1):
            submit_btn = gr.Button("Submit",variant="primary")

        with gr.Column(scale=1):
            clear_btn = gr.Button("Clear",variant="stop")
    
    
            
    text.submit(add_text, [chatbot, text], [chatbot,text]).then(bot, chatbot, chatbot)

    submit_btn.click(add_text, [chatbot, text], [chatbot,text]).then(bot, chatbot, chatbot)

    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=3)
    demo.launch()