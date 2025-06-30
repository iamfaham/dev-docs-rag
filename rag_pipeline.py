import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize Pinecone vectorstore
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

index_name = os.getenv("PINECONE_INDEX")
vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embedder,
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM setup
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Prompt template
template = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Helpful Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# RAG chain
def format_docs(docs):
    return "\n\n".join(f"{doc.metadata['title']}:\n{doc.page_content}" for doc in docs)


rag_chain = (
    RunnableLambda(
        lambda q: {
            "question": q,
            "context": format_docs(retriever.get_relevant_documents(q)),
        }
    )
    | prompt
    | llm
)

# Run it for local testing
if __name__ == "__main__":
    while True:
        query = input("\n Ask a question about React: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = rag_chain.invoke(query)
        print("\n🤖 Answer:\n", response.content)
