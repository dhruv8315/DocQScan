import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from backend.bot_conversation import conversation
from backend.script import process_pdf


def upload_pdf(file):
    process_pdf(file)

def add_text(history, text):
    if history is None:
        history = []

    history = history + [{"role": "user", "content": text}]
    return history, ""

def convert_gradio_history(history):
    
    lc_history = []

    for msg in history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))
    
    return lc_history

def bot(history):
    """user_msg = history[-1]["content"]

    lc_history = convert_gradio_history(history[:-1])"""

    raw_content = history[-1]["content"]

    if isinstance(raw_content, list):
        user_msg = raw_content[0]["text"]
    else:
        user_msg = raw_content
    
    lc_history = convert_gradio_history(history[:-1])
    
    qa = conversation()
    
    print("TYPE user_msg:", type(user_msg))
    print("VALUE user_msg:", user_msg)

    print("TYPE lc_history:", type(lc_history))
    print("TYPE first history element:", type(lc_history[0]) if lc_history else None)

    response = qa.invoke({
        "input": user_msg,
        "chat_history": lc_history
    })
    

    history[-1] = {"role": "assistant", "content": response["answer"]}

    return history
"""def bot(history):
    res = qa(
        {
            "question": history[-1][0],
            "chat_history": history[:-1]
        }

    )
    history[-1][1] = res
    return history
"""
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