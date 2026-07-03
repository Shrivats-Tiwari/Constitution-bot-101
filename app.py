import os
import time

import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# CONFIG
# =========================

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

st.set_page_config(
    page_title="Constitution RAG Chatbot",
    page_icon="📜",
    layout="centered",
)

st.title("📜 Constitution of India RAG Chatbot")
st.caption(
    "Ask questions about the Constitution of India using Retrieval-Augmented Generation (RAG)"
)

# =========================
# GEMINI SETUP
# =========================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel(GEMINI_MODEL)

# =========================
# LOAD VECTOR DATABASE
# =========================

@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )

    return retriever


retriever = load_rag()

# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================

question = st.chat_input(
    "Ask a constitutional question..."
)

if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching the Constitution..."):

            # -------------------------
            # Retrieval
            # -------------------------

            retrieval_start = time.perf_counter()

            docs = retriever.invoke(question)

            retrieval_seconds = (
                time.perf_counter() - retrieval_start
            )

            if docs:
                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )
            else:
                context = "No relevant constitutional text found."

            # -------------------------
            # Prompt
            # -------------------------

            prompt = f"""
You are a helpful assistant answering questions about the Constitution of India.

Use only the supplied context.

If the answer is not present in the context, clearly state that.

Context:
{context}

Question:
{question}

Answer:
"""

            # -------------------------
            # Generation
            # -------------------------

            generation_start = time.perf_counter()

            try:

                response = model.generate_content(
                    prompt
                )

                answer = (
                    response.text.strip()
                    if response.text
                    else "No answer returned."
                )

            except ResourceExhausted:

                answer = f"""
### Retrieved Constitutional Passage

Gemini is currently rate-limited.

Showing the most relevant constitutional text instead:

{context[:1500]}
"""

            except Exception as e:

                answer = f"""
### Retrieved Constitutional Passage

The language model is currently unavailable.

Showing the retrieved constitutional text:

{context[:1500]}

Error:
{e}
"""

            generation_seconds = (
                time.perf_counter() - generation_start
            )

        # -------------------------
        # Display Answer
        # -------------------------

        st.markdown(answer)

        st.caption(
            f"Retrieval: {retrieval_seconds:.2f}s | "
            f"Generation: {generation_seconds:.2f}s | "
            f"Model: {GEMINI_MODEL}"
        )

        with st.expander(
            "Retrieved Context Preview"
        ):
            st.text(context[:2000])

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )