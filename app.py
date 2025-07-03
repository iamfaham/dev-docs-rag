import gradio as gr
from rag_pipeline import rag_chain  # reuse from Step 3 in rag_pipeline.py


def chat_with_rag(message, history):
    if not message.strip():
        return history, ""

    try:
        response = rag_chain.invoke(message)

        # Check if response is too long and truncate if necessary
        max_display_length = 8000  # Reasonable limit for Gradio display
        if len(response) > max_display_length:
            truncated_response = (
                response[:max_display_length]
                + "\n\n... (response truncated due to length)"
            )
            response = truncated_response

        # Add the exchange to history in the correct format for messages type
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def clear_chat():
    """Clear the chat history"""
    return [], ""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .chatbot {
        max-height: 600px !important;
        overflow-y: auto !important;
    }
    .chatbot .message {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
    }
    .chatbot .user-message, .chatbot .bot-message {
        padding: 10px !important;
        margin: 5px 0 !important;
        border-radius: 8px !important;
    }
    .chatbot .bot-message {
        background-color: #f0f8ff !important;
        border-left: 4px solid #007acc !important;
    }
    .chatbot .user-message {
        background-color: #e6f3ff !important;
        border-left: 4px solid #28a745 !important;
    }
    .send-button {
        background-color: #007acc !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: background-color 0.3s !important;
    }
    .send-button:hover {
        background-color: #005a9e !important;
    }
    .clear-button {
        background-color: #dc3545 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        transition: background-color 0.3s !important;
    }
    .clear-button:hover {
        background-color: #c82333 !important;
    }
    .input-container {
        display: flex !important;
        gap: 10px !important;
        align-items: flex-end !important;
    }
    .textbox-container {
        flex: 1 !important;
    }
""",
) as demo:
    gr.Markdown("# 🤖 React Docs Assistant")
    gr.Markdown(
        "Ask questions about React documentation and get comprehensive answers."
    )

    # Chat history
    chatbot = gr.Chatbot(
        label="Chat History",
        height=500,  # Slightly reduced to make room for input area
        show_label=True,
        type="messages",  # Use the new messages format
    )

    # Input area with send button
    with gr.Row():
        with gr.Column(scale=4):
            textbox = gr.Textbox(
                placeholder="Ask a question about React... (Press Enter or click Send)",
                lines=2,  # Allow multiple lines for longer questions
                max_lines=5,
                label="Your Question",
                show_label=True,
            )
        with gr.Column(scale=1):
            send_button = gr.Button(
                "🚀 Send", variant="primary", size="lg", elem_classes=["send-button"]
            )

    # Control buttons
    with gr.Row():
        clear_button = gr.Button(
            "🗑️ Clear Chat", variant="secondary", elem_classes=["clear-button"]
        )

    # Example questions
    with gr.Accordion("Example Questions", open=False):
        gr.Markdown(
            """
        Try these example questions:
        - **What is React?**
        - **How do I use useState hook?**
        - **Explain React components**
        - **What are props in React?**
        - **How does React rendering work?**
        - **What are React Hooks?**
        - **How to handle events in React?**
        """
        )

    # Event handlers
    def send_message(message, history):
        return chat_with_rag(message, history)

    # Connect the send button
    send_button.click(
        fn=send_message,
        inputs=[textbox, chatbot],
        outputs=[chatbot, textbox],
        api_name="send",
    )

    # Connect Enter key in textbox
    textbox.submit(
        fn=send_message,
        inputs=[textbox, chatbot],
        outputs=[chatbot, textbox],
        api_name="send_enter",
    )

    # Connect clear button
    clear_button.click(
        fn=clear_chat, inputs=[], outputs=[chatbot, textbox], api_name="clear"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public link
        debug=True,  # Enable debug mode for better error messages
        show_error=True,
    )
