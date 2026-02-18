import gradio as gr
from backend.bot_conversation import conversation
from backend.script import process_pdf

qa = conversation()


def upload_pdf(file):
    process_pdf(file)

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    res = qa(
        {
            "question": history[-1][0],
            "chat_history": history[:-1]
        }

    )
    history[-1][1] = res
    return history

with gr.Blocks() as demo:

    with gr.Row(scale=3):
        file = gr.File(label="Upload PDF",file_types=[".pdf"],container=False)
        file.change(upload_pdf, file)

    chatbot = gr.Chatbot([],elem_id="chatbot",show_label=False,height=550)

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
    
    
            
    text.submit(add_text, [chatbot, text], chatbot).then(bot, chatbot, chatbot)

    submit_btn.click(add_text, [chatbot, text], chatbot).then(bot, chatbot, chatbot)

    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=3)
    demo.launch()