import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import json
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import re
from appwrite_service import appwrite_service

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def detect_device():
    """Detect the best available device for computation"""
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        device = "cpu"
        logging.info("💻 Using CPU for computation")
    return device


# Initialize device
device = detect_device()

# Initialize Pinecone vectorstore with GPU support
logging.info(f"🧠 Initializing embeddings model on {device.upper()}")
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": device},
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
    max_tokens=2000,
    temperature=0.7,
)

# Relevance check prompt template
relevance_template = """You are a helpful assistant that determines if a question is related to the available documentation.

Available Documentation Context:
{context}

Question: {question}

Instructions:
- Answer "YES" if the question is related to ANY topic, concept, feature, or technology mentioned in the documentation context above
- Answer "YES" if the question asks about general concepts that would be covered in this type of documentation
- Answer "NO" only if the question is clearly about a completely different technology, domain, or unrelated topic
- Be generous in your interpretation - if there's any reasonable chance the documentation could help answer the question, answer "YES"

Examples:
- For React documentation: Questions about hooks, components, JSX, state, props, lifecycle, etc. should be "YES"
- For Python documentation: Questions about syntax, libraries, functions, data types, etc. should be "YES"
- For any documentation: Questions about basic concepts of that technology should be "YES"

Answer with ONLY "YES" or "NO":"""

relevance_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=relevance_template,
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


# Load docs for BM25 from Appwrite instead of local JSON
def load_docs_from_appwrite(selected_url=None):
    """Load document chunks from Appwrite database for specific documentation"""
    try:
        logging.info(f"Loading document chunks from Appwrite for URL: {selected_url}")
        docs_json = appwrite_service.get_all_chunks(selected_url)

        if not docs_json:
            logging.warning(
                f"No chunks found in Appwrite database for URL: {selected_url}. This is normal if no documentation has been processed yet."
            )
            # Return empty list instead of raising error
            return []

        logging.info(
            f"Loaded {len(docs_json)} chunks from Appwrite for URL: {selected_url}"
        )
        return docs_json
    except Exception as e:
        logging.error(f"Error loading docs from Appwrite: {str(e)}")
        # Return empty list on error instead of raising
        return []


# Global variables for BM25
docs_json = None
bm25_corpus = None
bm25_titles = None
bm25 = None
current_url = None  # Track current URL to detect changes


def reset_bm25_data():
    """Reset BM25 data to force reinitialization"""
    global docs_json, bm25_corpus, bm25_titles, bm25, current_url
    docs_json = None
    bm25_corpus = None
    bm25_titles = None
    bm25 = None
    current_url = None
    logging.info("BM25 data reset")


def initialize_bm25(selected_url=None):
    """Initialize BM25 with document chunks from Appwrite for specific documentation"""
    global docs_json, bm25_corpus, bm25_titles, bm25, current_url

    # Reset if URL has changed
    if current_url != selected_url:
        logging.info(
            f"URL changed from {current_url} to {selected_url}, resetting BM25 data"
        )
        reset_bm25_data()
        current_url = selected_url

    if docs_json is None:
        docs_json = load_docs_from_appwrite(selected_url)

        if not docs_json:
            # If no chunks available, create empty BM25
            bm25_corpus = []
            bm25_titles = []
            bm25 = None  # Don't initialize BM25 with empty corpus
            logging.warning(
                f"BM25 initialized with no chunks for URL: {selected_url} - no documentation processed yet"
            )
        else:
            bm25_corpus = [doc["content"] for doc in docs_json]
            bm25_titles = [doc.get("title", "") for doc in docs_json]
            bm25 = BM25Okapi([doc.split() for doc in bm25_corpus])
            logging.info(
                f"BM25 initialized with {len(docs_json)} chunks for URL: {selected_url}"
            )


# Cross-encoder for re-ranking (kept on CPU as requested - no GPU acceleration for re-ranking)
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
cross_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)
logging.info(
    "🔄 Cross-encoder model initialized on CPU (re-ranking excluded from GPU acceleration)"
)


# Create context summary for relevance checking
def create_context_summary(selected_url=None):
    """Create a comprehensive summary of available context for relevance checking"""
    try:
        # Initialize BM25 if not already done
        initialize_bm25(selected_url)

        # Get unique titles from the corpus
        if bm25_titles:
            unique_titles = list(set(bm25_titles))

            # Create a more comprehensive context summary
            # Include more titles and also extract key topics from content
            context_parts = []

            # Add document titles (increase from 20 to 50 for better coverage)
            context_parts.append("Document titles:")
            context_parts.extend(unique_titles[:50])

            # Add key topics extracted from content
            if bm25_corpus:
                # Extract key terms from the first few documents
                key_terms = set()
                for doc_content in bm25_corpus[:100]:  # Check first 100 docs
                    # Extract important terms (simple approach)
                    words = doc_content.lower().split()
                    # Look for React-specific terms
                    react_terms = [
                        word
                        for word in words
                        if any(
                            term in word
                            for term in [
                                "hook",
                                "component",
                                "jsx",
                                "prop",
                                "state",
                                "effect",
                                "context",
                                "reducer",
                                "ref",
                                "memo",
                                "callback",
                                "usememo",
                                "usestate",
                                "useeffect",
                                "usecontext",
                                "usereducer",
                                "useref",
                                "usecallback",
                                "react",
                                "render",
                                "virtual",
                                "dom",
                                "lifecycle",
                            ]
                        )
                    ]
                    key_terms.update(react_terms[:10])  # Limit per document

                if key_terms:
                    context_parts.append("\nKey topics found:")
                    context_parts.extend(list(key_terms)[:30])  # Top 30 key terms

            # Add URL information for context
            if selected_url:
                context_parts.append(f"\nDocumentation source: {selected_url}")
                if "react" in selected_url.lower():
                    context_parts.append(
                        "This is React documentation covering components, hooks, JSX, state management, and React concepts."
                    )
                elif "python" in selected_url.lower():
                    context_parts.append(
                        "This is Python documentation covering language features, standard library, and Python concepts."
                    )
                elif "vue" in selected_url.lower():
                    context_parts.append(
                        "This is Vue.js documentation covering components, directives, and Vue concepts."
                    )
                # Add more URL-specific context as needed

            context_summary = "\n".join(context_parts)
        else:
            context_summary = "No documentation available yet"

        logging.info(f"Context summary created with {len(context_summary)} characters")
        return context_summary
    except Exception as e:
        logging.error(f"Error creating context summary: {str(e)}")
        return "Documentation topics"


# Hybrid retrieval function
def hybrid_retrieve(query, selected_url=None, dense_k=5, bm25_k=5, rerank_k=5):
    logging.info(f"Hybrid retrieval for query: {query} with URL: {selected_url}")

    # Initialize BM25 if not already done
    initialize_bm25(selected_url)

    # Dense retrieval
    dense_docs = retriever.get_relevant_documents(query)
    logging.info(f"Dense docs retrieved: {len(dense_docs)}")
    dense_set = set((d.metadata["title"], d.page_content) for d in dense_docs)

    # BM25 retrieval
    if (
        bm25_corpus and bm25 is not None
    ):  # Only if we have chunks and BM25 is initialized
        bm25_scores = bm25.get_scores(query.split())
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:bm25_k]
        bm25_docs = [
            Document(
                page_content=bm25_corpus[i],
                metadata={"title": bm25_titles[i]},
            )
            for i in bm25_indices
        ]
        logging.info(f"BM25 docs retrieved: {len(bm25_docs)}")
        bm25_set = set((d.metadata["title"], d.page_content) for d in bm25_docs)
    else:
        bm25_docs = []
        bm25_set = set()
        logging.info("No BM25 docs retrieved - no chunks available")

    # Merge and deduplicate
    all_docs = list(
        {(d[0], d[1]): d for d in list(dense_set) + list(bm25_set)}.values()
    )
    all_doc_objs = [
        Document(
            page_content=c,
            metadata={"title": t},
        )
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


# Relevance check function
def check_relevance(question, selected_url=None):
    """Check if the question is relevant to the available documentation"""
    try:
        logging.info(
            f"Checking relevance for question: {question} with URL: {selected_url}"
        )

        # First, check for obvious relevant keywords based on the URL
        question_lower = question.lower()
        if selected_url:
            url_lower = selected_url.lower()

            # Define technology-specific keywords
            tech_keywords = {
                "react": [
                    "hook",
                    "component",
                    "jsx",
                    "prop",
                    "state",
                    "effect",
                    "context",
                    "reducer",
                    "ref",
                    "memo",
                    "callback",
                    "render",
                    "virtual",
                    "dom",
                    "lifecycle",
                    "react",
                ],
                "python": [
                    "python",
                    "function",
                    "class",
                    "module",
                    "import",
                    "variable",
                    "list",
                    "dict",
                    "string",
                    "integer",
                    "loop",
                    "condition",
                    "exception",
                    "library",
                ],
                "vue": [
                    "vue",
                    "component",
                    "directive",
                    "template",
                    "computed",
                    "watch",
                    "method",
                    "prop",
                    "emit",
                    "slot",
                    "router",
                    "vuex",
                ],
                "node": [
                    "node",
                    "npm",
                    "express",
                    "server",
                    "module",
                    "require",
                    "async",
                    "callback",
                    "promise",
                    "stream",
                ],
                "django": [
                    "django",
                    "model",
                    "view",
                    "template",
                    "form",
                    "admin",
                    "url",
                    "middleware",
                    "orm",
                    "queryset",
                ],
                "docker": [
                    "docker",
                    "container",
                    "image",
                    "dockerfile",
                    "compose",
                    "volume",
                    "network",
                    "registry",
                ],
                "kubernetes": [
                    "kubernetes",
                    "pod",
                    "service",
                    "deployment",
                    "namespace",
                    "ingress",
                    "configmap",
                    "secret",
                ],
            }

            # Check if question contains relevant keywords for the current documentation
            for tech, keywords in tech_keywords.items():
                if tech in url_lower:
                    if any(keyword in question_lower for keyword in keywords):
                        logging.info(
                            f"Question contains relevant {tech} keywords - bypassing LLM relevance check"
                        )
                        return True

        # Create context summary
        context_summary = create_context_summary(selected_url)

        # Log the context summary for debugging
        logging.info(f"Context summary for relevance check: {context_summary[:500]}...")

        # Check relevance using LLM
        relevance_response = llm.invoke(
            relevance_prompt.format(context=context_summary, question=question)
        )

        # Parse the response
        response_text = relevance_response.content.strip().upper()
        is_relevant = "YES" in response_text

        logging.info(
            f"Relevance check result: {response_text} (Relevant: {is_relevant})"
        )

        # If LLM says NO but we have keyword matches, override to YES
        if not is_relevant and selected_url:
            url_lower = selected_url.lower()
            if "react" in url_lower and any(
                term in question_lower
                for term in ["hook", "component", "jsx", "state", "prop", "react"]
            ):
                logging.info(
                    "Overriding LLM relevance check - question contains React-specific terms"
                )
                return True
            elif "python" in url_lower and any(
                term in question_lower
                for term in ["python", "function", "class", "module"]
            ):
                logging.info(
                    "Overriding LLM relevance check - question contains Python-specific terms"
                )
                return True

        return is_relevant

    except Exception as e:
        logging.error(f"Error in relevance check: {str(e)}")
        # Default to relevant if check fails
        logging.info("Defaulting to relevant due to error")
        return True


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


def process_question_with_relevance_check(
    original_question, selected_url=None, debug=False
):
    try:
        logging.info(
            f"Processing question with relevance check: {original_question} for URL: {selected_url}"
        )

        # Step 1: Check if the question is relevant to the documentation
        is_relevant = check_relevance(original_question, selected_url)

        if debug:
            print(f"🔍 DEBUG: Question: {original_question}")
            print(f"🔍 DEBUG: URL: {selected_url}")
            print(f"🔍 DEBUG: Relevance check result: {is_relevant}")

        if not is_relevant:
            logging.info(
                f"Question not relevant to available documentation: {original_question}"
            )
            error_msg = f'No context provided for "{original_question}". This question doesn\'t appear to be related to the documentation that has been processed. Please ask a question about the documentation topics that are available.'

            if debug:
                print(f"🔍 DEBUG: Returning relevance error: {error_msg}")
                # Also show the context that was used for relevance check
                context = create_context_summary(selected_url)
                print(f"🔍 DEBUG: Context used for relevance check: {context[:500]}...")

            return error_msg

        # Step 2: If relevant, proceed with decomposition
        sub_questions = decompose_question(original_question)
        logging.info(f"Sub-questions: {sub_questions}")

        if debug:
            print(f"🔍 DEBUG: Sub-questions: {sub_questions}")

        # Step 3: Get answers for each sub-question
        sub_answers = []
        for i, sub_q in enumerate(sub_questions, 1):
            logging.info(f"Processing sub-question {i}: {sub_q}")

            # Retrieve context for this sub-question
            context = format_docs(hybrid_retrieve(sub_q, selected_url))
            logging.info(f"Context length for sub-question {i}: {len(context)}")

            if debug:
                print(f"🔍 DEBUG: Sub-question {i}: {sub_q}")
                print(f"🔍 DEBUG: Context length: {len(context)}")

            # Get answer for this sub-question
            sub_answer = llm.invoke(prompt.format(context=context, question=sub_q))
            logging.info(f"Sub-answer {i}: {sub_answer.content[:100]}...")
            sub_answers.append(f"{i}. {sub_q}\nAnswer: {sub_answer.content}")

        # Step 4: Synthesize the final answer
        sub_answers_text = "\n\n".join(sub_answers)
        logging.info(f"Sub-answers text length: {len(sub_answers_text)}")

        final_answer = llm.invoke(
            synthesis_prompt.format(
                original_question=original_question, sub_answers=sub_answers_text
            )
        )

        logging.info(f"Final answer: {final_answer.content[:100]}...")

        if debug:
            print(f"🔍 DEBUG: Final answer length: {len(final_answer.content)}")

        return final_answer.content

    except Exception as e:
        logging.error(f"Error in process_question_with_relevance_check: {str(e)}")
        return f"Error processing question: {str(e)}"


# Enhanced RAG chain with relevance check
def create_rag_chain(selected_url=None, debug=False):
    """Create a RAG chain for the selected documentation"""

    def process_with_url(question):
        return process_question_with_relevance_check(question, selected_url, debug)

    return RunnableLambda(process_with_url)


# Default RAG chain (for backward compatibility)
rag_chain = create_rag_chain()

# Run it for local testing
if __name__ == "__main__":
    while True:
        query = input("\n Ask a question about the documentation: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = rag_chain.invoke(query)
        print("\n🤖 Answer:\n", response)
