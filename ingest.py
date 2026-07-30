import os

from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)

from utils import (
    load_pdf,
    split_documents,
    get_embedding_model,
    add_pdf_metadata,
   
)

import config


# Ingest PDF into Qdrant


def ingest_pdf(pdf_path: str):

    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Loading PDF...")

    documents = load_pdf(pdf_path)
    documents = add_pdf_metadata(documents)
    for doc in documents[:30]:

     print(
        doc.metadata.get("page"),
        doc.metadata.get("chapter_number"),
        doc.metadata.get("chapter_name")
    )

    print(f"Loaded {len(documents)} pages")

    print("Splitting document...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")

    embeddings = get_embedding_model()
    sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

    print("Connecting to Qdrant...")

    print("Creating vector database...")

    

    vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    url=config.Qdrant_url,
    collection_name=config.COLLECTION_NAME,
    
)
    
    print("===================================")
    print("PDF successfully indexed!")
    print(f"Collection: {config.COLLECTION_NAME}")
    print(f"Chunks stored: {len(chunks)}")
    print("===================================")

    return vectorstore



# Main


if __name__ == "__main__":

    pdf_path = "pdf/pakistanlaw.pdf"

    ingest_pdf(pdf_path)