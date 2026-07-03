import os
import time

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

print("Loading vector database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print("Constitution RAG ready!")

while True:

    question = input("\nAsk a question (or type quit): ")

    if question.lower() == "quit":
        break

    print("\nRetrieving relevant chunks...")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k":8}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    print("\nRetrieved Context Preview:")
    print(context[:1000])

    prompt = f"""
You are a helpful assistant answering questions about the Constitution of India.

Use only the supplied context.
If the answer cannot be found in the context, say so.

Context:
{context}

Question:
{question}

Answer:
"""

    print("\nGenerating answer...")

    MAX_RETRIES = 5

    for attempt in range(MAX_RETRIES):

        try:
            response = model.generate_content(prompt)

            print("\nAnswer:\n")
            print(response.text)

            break

        except ResourceExhausted:

            wait_time = 25

            print(
                f"\nRate limit reached. "
                f"Waiting {wait_time} seconds before retrying..."
            )

            time.sleep(wait_time)

        except Exception as e:

            print(f"\nUnexpected error: {e}")

            break