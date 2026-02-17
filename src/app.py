import gradio as gr
from backend.bot_conversation import conversation

qa = conversation()


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

    chatbot = gr.Chatbot([],elem_id="chatbot",show_label=False,height=750)

    with gr.Row():
        with gr.Column(scale=0.80):
            text = gr.Textbox(
                show_label=False, 
                placeholder="Ask Anything !",
                container=False
                )
        
        with gr.Column(scale=0.10):
            submit_btn = gr.Button("Submit",variant="primary")

        with gr.Column(scale=0.10):
            clear_btn = gr.Button("Clear",variant="stop")
            
    text.submit(add_text, [chatbot, text], chatbot).then(bot, chatbot, chatbot)

    submit_btn.click(add_text, [chatbot, text], chatbot).then(bot, chatbot, chatbot)

    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=3)
    demo.launch()