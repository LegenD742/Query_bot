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

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vectorstore


def get_ingested_files(vectorstore):

    data = vectorstore.get(
        include=["metadatas"]
    )

    ingested_files = set()

    for metadata in data["metadatas"]:

        if metadata and "source" in metadata:
            ingested_files.add(metadata["source"])

    return ingested_files


# -----------------------------
# Ingest PDFs
# -----------------------------

def ingest_pdfs():

    print("\nStarting PDF ingestion...\n")

    vectorstore = get_vectorstore()

    # PDFs already present in Chroma
    ingested_files = get_ingested_files(vectorstore)

    print(
        f"Already ingested: {len(ingested_files)} PDF(s)\n"
    )

    # Find all PDFs
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:

        print("No PDF files found in data/")

        return

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=200
    )

    new_files = 0
    new_chunks = 0

    # Process each PDF
    for pdf_file in pdf_files:

        filename = pdf_file.name

        # --------------------------------
        # Skip already ingested documents
        # --------------------------------

        if filename in ingested_files:

            print(f"SKIP  → {filename}")

            continue

        print(f"INGEST → {filename}")

        # Load PDF
        loader = PyPDFLoader(
            str(pdf_file)
        )

        documents = loader.load()

        # Split into chunks
        chunks = text_splitter.split_documents(
            documents
        )

        # Add useful metadata
        for chunk in chunks:

            chunk.metadata["source"] = filename

        # Add to Chroma
        vectorstore.add_documents(
            documents=chunks
        )

        print(
            f"         {len(documents)} pages"
        )

        print(
            f"         {len(chunks)} chunks"
        )

        new_files += 1
        new_chunks += len(chunks)

    # Summary
    print("\n-----------------------------")
    print("Ingestion complete")
    print("-----------------------------")

    print(
        f"New PDFs:    {new_files}"
    )

    print(
        f"New chunks:  {new_chunks}"
    )



if __name__ == "__main__":

    ingest_pdfs()