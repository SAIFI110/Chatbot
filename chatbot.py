from langchain_qdrant import QdrantVectorStore, RetrievalMode

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains import create_retrieval_chain

from utils import get_embedding_model

from utils import get_qdrant_client

client = get_qdrant_client()

import config


from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)




# Load Embedding Model

embedding_model = get_embedding_model()
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)



# Connect Qdrant

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=config.COLLECTION_NAME,
    embedding=embedding_model,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)
# Retriever



retriever = vectorstore.as_retriever(

   search_kwargs={
    "k": 10
}

)
# LLM


llm = ChatOpenAI(

    model=config.LLM_MODEL,

    temperature=0

)


# Prompt


prompt = ChatPromptTemplate.from_template(
"""
You are a legal AI assistant for Pakistan Penal Code 1860.

Answer ONLY from the provided context.

Rules:
- Use only relevant legal text from the context.
- Prefer actual section content over references, footnotes, amendments, or table of contents.
- If a section number is only mentioned but its actual text is missing, do not answer from it.
- Combine information from multiple chunks when required.
- Do not use outside knowledge or make assumptions.
- If the answer is not found, reply exactly:
"I don't know the answer based on the provided document."

Provide answers in a clear legal format:

Section:
Offence:
Punishment:
Explanation:

Context:
{context}

Question:
{input}

Answer:
"""
)
# Document Chain


document_chain = create_stuff_documents_chain(

    llm,

    prompt

)



# Retrieval Chain


rag_chain = create_retrieval_chain(

    retriever,

    document_chain

)


# Ask Question Function


def ask_question(question: str):

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    print("\n====================")
    print("QUESTION:", question)

    print("\nANSWER:")
    print(response["answer"])

    print("\nRETRIEVED CONTEXT:")

    for i, document in enumerate(response["context"]):

        print("\n--- Chunk", i + 1, "---")
        print(document.page_content[:300])
        print(document.metadata)

    print("====================\n")

    sources = []

    for document in response["context"]:

        metadata = document.metadata

        sources.append(
    {
        "source": metadata.get("source"),
        "page": metadata.get("page", 0),
        "chapter_number": metadata.get("chapter_number"),
        "chapter_name": metadata.get("chapter_name"),
    }
)

    return {
        "answer": response["answer"],
        "sources": sources,
        "documents": response["context"],
    }