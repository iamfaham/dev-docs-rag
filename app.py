import gradio as gr
from rag_pipeline import rag_chain  # reuse from Step 3


def chat_with_rag(message, history):
    response = rag_chain.invoke(message)
    return response.content


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 React Docs Assistant")
    gr.ChatInterface(
        fn=chat_with_rag,
        chatbot=gr.Chatbot(label="Assistant"),
        textbox=gr.Textbox(placeholder="Ask a question about React...", scale=7),
        title="React Docs RAG Assistant",
    )

if __name__ == "__main__":
    demo.launch()
