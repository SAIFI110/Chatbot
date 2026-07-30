from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
import re

import config
#extract metadata
# load PDF

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


def add_pdf_metadata(documents):

    current_chapter_number = None
    current_chapter_name = None


    for doc in documents:

        text = doc.page_content


        # Normalize spaces
        clean_text = " ".join(text.split())


        # Detect chapter heading
        match = re.search(
            r"CHAPTER\s*([IVXLCDM]+|\d+)\s*(?:OF)?\s*([A-Z][A-Z\s]+)",
            clean_text
        )


        if match:

            current_chapter_number = match.group(1)

            current_chapter_name = match.group(2).strip()


        doc.metadata["page"] = doc.metadata.get("page",0)+1

        doc.metadata["chapter_number"] = current_chapter_number

        doc.metadata["chapter_name"] = current_chapter_name


    return documents
# Split PDF


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)



# OpenAI Embeddings


def get_embedding_model():

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY
    )

    return embeddings


   # Qdrant Client


from qdrant_client import QdrantClient

def get_qdrant_client():

    client = QdrantClient(
        url=config.Qdrant_url,
    )

    return client