from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "college_policies"

EMBEDDING_MODEL = "nomic-embed-text"


def get_retriever():

    # Same embedding model used during ingestion
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Connect to existing ChromaDB
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Return top 4 most similar chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever