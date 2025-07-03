import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import json
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import re

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    max_tokens=2000,  # Limit response length to prevent extremely long outputs
    temperature=0.7,  # Add some creativity while keeping responses focused
)

# Question decomposition prompt template
decomposition_template = """Break down the following question into exactly 4 sub-questions that would help provide a comprehensive answer. 
Each sub-question should focus on a different aspect of the main question.

Original Question: {question}

Please provide exactly 4 sub-questions, one per line, starting with numbers 1-4:

1. [First sub-question]
2. [Second sub-question] 
3. [Third sub-question]
4. [Fourth sub-question]

Make sure each sub-question is specific and focused on a different aspect of the original question."""

decomposition_prompt = PromptTemplate(
    input_variables=["question"],
    template=decomposition_template,
)

# Answer synthesis prompt template
synthesis_template = """You are a helpful assistant. Based on the answers to the sub-questions below, provide a comprehensive but concise answer to the original question.

Original Question: {original_question}

Sub-questions and their answers:
{sub_answers}

Please synthesize these answers into a clear, well-structured response that directly addresses the original question. 
Keep the response focused and avoid unnecessary repetition. If any sub-question couldn't be answered with the available context, mention that briefly.
Include relevant code examples where applicable, but keep them concise."""

synthesis_prompt = PromptTemplate(
    input_variables=["original_question", "sub_answers"],
    template=synthesis_template,
)

# Individual answer prompt template
template = """You are a helpful assistant. Answer the question using ONLY the context below. Also add a code example if applicable.
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

# Load docs for BM25
with open("react_docs_chunks.json", "r", encoding="utf-8") as f:
    docs_json = json.load(f)

bm25_corpus = [doc["content"] for doc in docs_json]
bm25_titles = [doc.get("title", "") for doc in docs_json]
bm25 = BM25Okapi([doc.split() for doc in bm25_corpus])

# Cross-encoder for re-ranking
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
cross_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)


# Hybrid retrieval function
def hybrid_retrieve(query, dense_k=5, bm25_k=5, rerank_k=5):
    logging.info(f"Hybrid retrieval for query: {query}")
    # Dense retrieval
    dense_docs = retriever.get_relevant_documents(query)
    logging.info(f"Dense docs retrieved: {len(dense_docs)}")
    dense_set = set((d.metadata["title"], d.page_content) for d in dense_docs)

    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:bm25_k]
    bm25_docs = [
        type(
            "Doc",
            (),
            {"metadata": {"title": bm25_titles[i]}, "page_content": bm25_corpus[i]},
        )
        for i in bm25_indices
    ]
    logging.info(f"BM25 docs retrieved: {len(bm25_docs)}")
    bm25_set = set((d.metadata["title"], d.page_content) for d in bm25_docs)

    # Merge and deduplicate
    all_docs = list(
        {(d[0], d[1]): d for d in list(dense_set) + list(bm25_set)}.values()
    )
    all_doc_objs = [
        type("Doc", (), {"metadata": {"title": t}, "page_content": c})
        for t, c in all_docs
    ]
    logging.info(f"Total unique docs before re-ranking: {len(all_doc_objs)}")

    # Re-rank with cross-encoder
    pairs = [(query, doc.page_content) for doc in all_doc_objs]
    inputs = cross_tokenizer.batch_encode_plus(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        scores = cross_model(**inputs).logits.squeeze().cpu().numpy()
    ranked = sorted(zip(all_doc_objs, scores), key=lambda x: x[1], reverse=True)[
        :rerank_k
    ]
    logging.info(f"Docs after re-ranking: {len(ranked)}")
    return [doc for doc, _ in ranked]


# Question decomposition function
def decompose_question(question):
    try:
        logging.info(f"Decomposing question: {question}")
        decomposition_response = llm.invoke(
            decomposition_prompt.format(question=question)
        )
        logging.info(
            f"Decomposition response: {decomposition_response.content[:200]}..."
        )

        # Extract sub-questions from the response
        content = decomposition_response.content
        sub_questions = []

        # Use regex to extract numbered questions
        pattern = r"\d+\.\s*(.+)"
        matches = re.findall(pattern, content, re.MULTILINE)
        logging.info(f"Regex matches: {matches}")

        for match in matches[:4]:  # Take first 4 matches
            sub_question = match.strip()
            if sub_question:
                sub_questions.append(sub_question)

        # If we don't get exactly 4 questions, create variations
        while len(sub_questions) < 4:
            sub_questions.append(f"Additional aspect of: {question}")

        logging.info(f"Decomposed into {len(sub_questions)} sub-questions")
        return sub_questions[:4]
    except Exception as e:
        logging.error(f"Error in decompose_question: {str(e)}")
        # Fallback to simple variations
        return [
            f"What is {question}?",
            f"How does {question} work?",
            f"When to use {question}?",
            f"Examples of {question}",
        ]


# RAG chain
def format_docs(docs):
    logging.info(f"Formatting {len(docs)} docs for LLM context.")
    return "\n\n".join(f"{doc.metadata['title']}:\n{doc.page_content}" for doc in docs)


def process_question_with_decomposition(original_question):
    try:
        logging.info(f"Processing question with decomposition: {original_question}")

        # Step 1: Decompose the question
        sub_questions = decompose_question(original_question)
        logging.info(f"Sub-questions: {sub_questions}")

        # Step 2: Get answers for each sub-question
        sub_answers = []
        for i, sub_q in enumerate(sub_questions, 1):
            logging.info(f"Processing sub-question {i}: {sub_q}")

            # Retrieve context for this sub-question
            context = format_docs(hybrid_retrieve(sub_q))
            logging.info(f"Context length for sub-question {i}: {len(context)}")

            # Get answer for this sub-question
            sub_answer = llm.invoke(prompt.format(context=context, question=sub_q))
            logging.info(f"Sub-answer {i}: {sub_answer.content[:100]}...")
            sub_answers.append(f"{i}. {sub_q}\nAnswer: {sub_answer.content}")

        # Step 3: Synthesize the final answer
        sub_answers_text = "\n\n".join(sub_answers)
        logging.info(f"Sub-answers text length: {len(sub_answers_text)}")

        final_answer = llm.invoke(
            synthesis_prompt.format(
                original_question=original_question, sub_answers=sub_answers_text
            )
        )

        logging.info(f"Final answer: {final_answer.content[:100]}...")
        return final_answer.content

    except Exception as e:
        logging.error(f"Error in process_question_with_decomposition: {str(e)}")
        return f"Error processing question: {str(e)}"


# Enhanced RAG chain with decomposition
rag_chain = RunnableLambda(process_question_with_decomposition)

# Run it for local testing
if __name__ == "__main__":
    while True:
        query = input("\n Ask a question about React: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = rag_chain.invoke(query)
        print("\n🤖 Answer:\n", response)
