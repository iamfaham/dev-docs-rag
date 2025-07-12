import os
import gradio as gr
from rag_pipeline import create_rag_chain
import time
import logging
from appwrite_service import appwrite_service

# Check if running on Hugging Face Spaces
IS_HF_SPACES = os.getenv("SPACE_ID") is not None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Predefined documentation sets
PREDEFINED_DOCS = {
    "React": {
        "name": "React Documentation",
        "url": "https://react.dev/learn",
        "description": "Official React documentation including hooks, components, and best practices",
        "category": "Frontend Framework",
    },
    "Go": {
        "name": "Go Documentation",
        "url": "https://go.dev/doc/",
        "description": "Official Go documentation including language features, standard library, and tutorials",
        "category": "Programming Language",
    },
    "Python": {
        "name": "Python Documentation",
        "url": "https://docs.python.org/3/",
        "description": "Official Python documentation covering language features, standard library, and tutorials",
        "category": "Programming Language",
    },
    "Node.js": {
        "name": "Node.js Documentation",
        "url": "https://nodejs.org/en/docs/",
        "description": "Node.js runtime documentation including APIs, modules, and development guides",
        "category": "Runtime Environment",
    },
    "Vue.js": {
        "name": "Vue.js Documentation",
        "url": "https://vuejs.org/guide/",
        "description": "Vue.js framework documentation with composition API, components, and routing",
        "category": "Frontend Framework",
    },
    "Django": {
        "name": "Django Documentation",
        "url": "https://docs.djangoproject.com/en/stable/",
        "description": "Django web framework documentation including models, views, and deployment",
        "category": "Backend Framework",
    },
    "FastAPI": {
        "name": "FastAPI Documentation",
        "url": "https://fastapi.tiangolo.com/",
        "description": "FastAPI framework documentation with automatic API documentation and validation",
        "category": "Backend Framework",
    },
    "Docker": {
        "name": "Docker Documentation",
        "url": "https://docs.docker.com/",
        "description": "Docker containerization platform documentation including images, containers, and orchestration",
        "category": "DevOps",
    },
    "Kubernetes": {
        "name": "Kubernetes Documentation",
        "url": "https://kubernetes.io/docs/",
        "description": "Kubernetes orchestration platform documentation including pods, services, and deployment",
        "category": "DevOps",
    },
    "MongoDB": {
        "name": "MongoDB Documentation",
        "url": "https://docs.mongodb.com/",
        "description": "MongoDB NoSQL database documentation including CRUD operations and aggregation",
        "category": "Database",
    },
    "PostgreSQL": {
        "name": "PostgreSQL Documentation",
        "url": "https://www.postgresql.org/docs/",
        "description": "PostgreSQL relational database documentation including SQL features and administration",
        "category": "Database",
    },
}

# Global variable to track selected documentation
selected_docs = {"key": None, "name": None, "url": None}


def select_documentation(doc_key):
    """Select a predefined documentation set"""
    global selected_docs

    if doc_key not in PREDEFINED_DOCS:
        return "❌ Invalid documentation selection"

    doc_info = PREDEFINED_DOCS[doc_key]
    selected_docs["key"] = doc_key
    selected_docs["name"] = doc_info["name"]
    selected_docs["url"] = doc_info["url"]

    # Check detailed status
    status = get_detailed_status(doc_info["url"])

    if "✅ Available" in status:
        return f"✅ {doc_info['name']} is ready! You can now ask questions about it."
    elif "⚠️" in status:
        return f"⚠️ {doc_info['name']} selected but not fully available. Contact administrator."
    else:
        return f"❌ {doc_info['name']} is not available. Contact administrator."


def chat_with_rag(message, history):
    """Chat with RAG system"""
    global selected_docs

    if not message.strip():
        return history, ""

    # Check if documentation is selected and processed
    if not selected_docs["key"]:
        error_msg = "❌ Please select a documentation set first. Go to the 'Select Documentation' tab."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

    # Check if documentation is fully processed and available for chat
    is_fully_processed = appwrite_service.is_fully_processed(selected_docs["url"])

    if not is_fully_processed:
        error_msg = f"❌ {selected_docs['name']} is not available for chat. Please contact the administrator to make this documentation available."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

    try:
        # Create RAG chain for the selected documentation
        rag_chain = create_rag_chain(selected_docs["url"])
        response = rag_chain.invoke(message)

        # Check if response is too long and truncate if necessary
        max_display_length = 8000
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


def get_detailed_status(url):
    """Get detailed status of documentation availability"""
    if not url:
        return "❌ No URL provided"

    try:
        # Check if fully processed (has completion status)
        is_fully_processed = appwrite_service.is_fully_processed(url)

        if is_fully_processed:
            return "✅ Available for Chat"
        else:
            return "❌ Not Available - Contact Admin"
    except Exception as e:
        return f"❌ Error checking status: {str(e)}"


def get_current_selection():
    """Get current documentation selection info with detailed status"""
    global selected_docs

    if selected_docs["key"]:
        doc_info = PREDEFINED_DOCS[selected_docs["key"]]
        status = get_detailed_status(selected_docs["url"])
        return f"📚 {doc_info['name']}\n📖 {doc_info['description']}\n🔗 {doc_info['url']}\n\nStatus: {status}"
    else:
        return "❌ No documentation selected. Please select a documentation set from the list above."


# Create the Gradio interface
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
    .select-button {
        background-color: #17a2b8 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        transition: background-color 0.3s !important;
    }
    .select-button:hover {
        background-color: #138496 !important;
    }
    .doc-selector {
        background-color: #f8f9fa !important;
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin-bottom: 20px !important;
    }
    .doc-selector:hover {
        border-color: #007acc !important;
        background-color: #e6f3ff !important;
    }
""",
) as demo:
    gr.Markdown("# 🤖 Documentation Assistant")
    gr.Markdown("Select documentation and start chatting!")

    # Documentation Selection Section (Small section at top)
    with gr.Group(elem_classes=["doc-selector"]):
        gr.Markdown("### 📚 Select Documentation")

        # Get available documentation from database
        def get_available_docs():
            """Get only documentation that is available in the database"""
            available_docs = {}
            available_options = []

            for key, doc_info in PREDEFINED_DOCS.items():
                if appwrite_service.is_fully_processed(doc_info["url"]):
                    available_docs[key] = doc_info
                    available_options.append(f"{doc_info['name']} - {doc_info['url']}")

            return available_docs, available_options

        # Get available documentation
        available_docs, doc_options = get_available_docs()
        doc_keys = list(available_docs.keys())

        if not available_docs:
            gr.Markdown("❌ **No documentation is currently available.**")
            gr.Markdown("Please contact the administrator to process documentation.")
        else:
            doc_dropdown = gr.Dropdown(
                choices=doc_options,
                label="Choose Documentation",
                value=None,
                interactive=True,
            )

            # Current selection display
            current_selection = gr.Textbox(
                label="Selected Documentation",
                interactive=False,
                value="No documentation selected",
                lines=2,
            )

    # Chat Interface (Main section)
    if available_docs:
        gr.Markdown("### 💬 Chat with Documentation")

        # Chat history
        chatbot = gr.Chatbot(
            label="Chat History",
            height=500,
            show_label=True,
            type="messages",
        )

        # Input area with send button
        with gr.Row():
            with gr.Column(scale=4):
                textbox = gr.Textbox(
                    placeholder="Ask a question about the documentation... (Press Enter or click Send)",
                    lines=2,
                    max_lines=5,
                    label="Your Question",
                    show_label=True,
                )
            with gr.Column(scale=1):
                send_button = gr.Button(
                    "🚀 Send",
                    variant="primary",
                    size="lg",
                    elem_classes=["send-button"],
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
            Try these example questions after selecting documentation:
            - **What is the main concept?**
            - **How do I get started?**
            - **What are the key features?**
            - **Show me an example**
            - **What are the best practices?**
            """
            )

        # Event handlers
        def select_doc_from_dropdown(choice):
            """Handle documentation selection from dropdown"""
            if not choice:
                return "No documentation selected"

            # Find the key for the selected option
            selected_index = doc_options.index(choice)
            selected_key = doc_keys[selected_index]

            # Call the existing select_documentation function
            return select_documentation(selected_key)

        def send_message(message, history):
            return chat_with_rag(message, history)

        def update_selection():
            return get_current_selection()

        # Connect the dropdown
        doc_dropdown.change(
            fn=select_doc_from_dropdown,
            inputs=[doc_dropdown],
            outputs=[current_selection],
        )

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

        # Update selection info on load
        demo.load(
            fn=update_selection,
            inputs=[],
            outputs=[current_selection],
        )
    else:
        gr.Markdown("### 💬 Chat Interface")
        gr.Markdown("**No documentation is available for chat.**")
        gr.Markdown("Please contact the administrator to process documentation first.")

if __name__ == "__main__":
    demo.launch(
        debug=False,
        show_error=True,
    )
