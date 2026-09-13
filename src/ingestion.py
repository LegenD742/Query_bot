from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "college_policies"

EMBEDDING_MODEL = "nomic-embed-text"


def get_vectorstore():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


def get_ingested_files(vectorstore):

    # Get all existing metadata from Chroma
    data = vectorstore.get()

    ingested_files = set()

    for metadata in data["metadatas"]:

        if metadata and "source" in metadata:
            ingested_files.add(metadata["source"])

    return ingested_files


def ingest():

    print("Starting ingestion...")

    vectorstore = get_vectorstore()

    # Files already present in Chroma
    ingested_files = get_ingested_files(vectorstore)

    print(f"Already ingested: {len(ingested_files)} PDF(s)")

    # Find PDFs
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=230
    )

    total_new_chunks = 0

    for pdf_file in pdf_files:

        filename = pdf_file.name

        # Skip already ingested PDFs
        if filename in ingested_files:

            print(f"Skipping: {filename}")

            continue

        print(f"Processing: {filename}")

        # Load PDF
        loader = PyPDFLoader(str(pdf_file))

        documents = loader.load()

        # Split into chunks
        chunks = text_splitter.split_documents(documents)

        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = filename

        # Add to ChromaDB
        vectorstore.add_documents(chunks)

        total_new_chunks += len(chunks)

        print(
            f"Added {len(chunks)} chunks from {filename}"
        )

    print("\nIngestion complete!")
    print(f"New chunks added: {total_new_chunks}")


if __name__ == "__main__":
    ingest()