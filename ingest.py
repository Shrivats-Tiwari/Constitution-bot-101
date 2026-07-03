from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Loading text file...")

with open("constitution.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded {len(text)} characters")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_texts(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created successfully!")