import gradio as gr
from backend.script import build_qa_system, answer_question

with gr.Blocks() as demo:

    qa_state = gr.State() # Create a state variable to store the state of the application, which can be used to manage and share data across different components and functions within the Gradio interface.
    with gr.Row():
        with gr.Column():
            
            file_input = gr.File(label="Upload a pdf file", file_types=[".pdf"]) # Create a file input component that allows users to upload PDF files, with a label "Upload a pdf file" and restricts the accepted file types to ".pdf".
            
            text_input = gr.Textbox(
                show_label=False, 
                placeholder="Ask Anything !"
                ) # Create a text input component with a label "Ask Anything !" and a placeholder text "Chat" to allow users to input their questions or messages.
            
            submit_btn = gr.Button("Submit") # Create a submit button with the label "Submit"

    with gr.Row():
        output = gr.Textbox(label="Result") # Create a textbox component to display the output result, with a label "Result"
    
    file_input.upload(
        fn=build_qa_system, 
        inputs=file_input, 
        outputs=qa_state
        ) # Set up an event listener for the file input component that triggers the `process_pdf` function when a file is uploaded. The function takes the uploaded file and the text input as arguments and outputs the result to the output textbox."""

    
    submit_btn.click(
        fn=answer_question, 
        inputs=[qa_state, text_input], 
        outputs=output
        ) # Set up an event listener for the submit button that triggers the `process_pdf` function when the button is clicked. The function takes the uploaded file and the text input as arguments and outputs the result to the output textbox.

demo.launch()