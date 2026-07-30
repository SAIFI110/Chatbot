from chatbot import retriever
from sentence_transformers import CrossEncoder

# Load once
reranker = CrossEncoder("BAAI/bge-reranker-base")


def retrieve_documents(question):

    # Step 1
    docs = retriever.invoke(question)

    # Step 2
    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]

    # Step 3
    scores = reranker.predict(pairs)

    # Step 4
    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    # Step 5
    final_docs = [
        doc
        for score, doc in ranked_docs[:5]
    ]

    return final_docs