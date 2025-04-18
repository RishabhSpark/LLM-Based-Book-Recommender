import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def build_vectorstore(csv_path: str, description_txt_path: str, persist_directory: str = "chroma_db") -> Chroma:
    """
    Builds and persists a Chroma vector database from book descriptions.

    Args:
        csv_path (str): Path to the cleaned book CSV.
        description_txt_path (str): Where to save tagged descriptions as .txt
        persist_directory (str): Directory to persist the Chroma DB.

    Returns:
        Chroma: The built vectorstore object.
    """
    load_dotenv()
    books = pd.read_csv(csv_path)
    books['tagged_description'].to_csv(
        description_txt_path, sep='\n', index=False, header=False
    )

    loader = TextLoader(description_txt_path, encoding='utf-8')
    raw_documents = loader.load()

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db_books = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return db_books
