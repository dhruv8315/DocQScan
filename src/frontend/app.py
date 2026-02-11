import gradio as gr
from PyPDF2 import PdfReader
from typing_extensions import Concatenate

def process_pdf(file_obj, text_input=""):
    
    file_path = file_obj.name  #Access the file path using file_obj
    reader = PdfReader(file_path) # Use PdfReader to read the PDF file and store the content in a reader

    raw_text = '' # Initialize an empty string to store the extracted text from the PDF

    for i, pages in enumerate(reader.pages): # Loop through each page in the PDF and extract the text content
        content = pages.extract_text()
        if content: # Check if the content is not empty before appending it to the raw_text variable
            raw_text += content


"""
This code defines a simple function `greet` that takes a name as input and returns a greeting message. The `gr.Interface` is used to create a web interface for this function, where users can input their name and receive the greeting. The `api_name` parameter allows the function to be accessed via an API endpoint named "predict". Finally, `demo.launch(share=True)` starts the interface and allows it to be shared publicly.
"""
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            
            file_input = gr.File(label="Upload a pdf file", file_types=[".pdf"]) # Create a file input component that allows users to upload PDF files, with a label "Upload a pdf file" and restricts the accepted file types to ".pdf".
            
            text_input = gr.Textbox(show_label=False, placeholder="Ask Anything !") # Create a text input component with a label "Ask Anything !" and a placeholder text "Chat" to allow users to input their questions or messages.
            
            submit_btn = gr.Button("Submit") # Create a submit button with the label "Submit"

    with gr.Row():
        output = gr.Textbox(label="Result") # Create a textbox component to display the output result, with a label "Result"
    
    """file_input.upload(fn=process_pdf, inputs=[file_input, text_input], outputs=output) # Set up an event listener for the file input component that triggers the `process_pdf` function when a file is uploaded. The function takes the uploaded file and the text input as arguments and outputs the result to the output textbox."""

    
    submit_btn.click(fn=process_pdf, inputs=[file_input, text_input], outputs=output) # Set up an event listener for the submit button that triggers the `process_pdf` function when the button is clicked. The function takes the uploaded file and the text input as arguments and outputs the result to the output textbox.

demo.launch()