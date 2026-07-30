from langchain_openai import ChatOpenAI



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def generate_answer(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""You are a legal AI assistant specializing in the Pakistan Penal Code 1860.

Instructions:
1. Answer ONLY from the provided context.

The wording of the user's question does NOT need to exactly match the wording in the context.

If the retrieved context clearly discusses the same legal subject,
use it to answer.

Infer the most relevant section from the context.
If the user's question uses non-legal or ambiguous language,
identify the closest relevant legal offence from the provided context.

If multiple legal interpretations are possible,
briefly state the ambiguity and answer using the closest matching provision.
Do not invent offences that are not supported by the context.
Only reply "I can't help with that."
when NONE of the retrieved passages are relevant.
Do NOT use outside legal knowledge or assumptions.

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    # Extract and deduplicate source metadata
    seen_sources = set()
    sources = []

    for doc in documents:
        meta = doc.metadata
        source_key = (
            meta.get("source"),
            meta.get("page"),
            meta.get("chapter_number"),
            meta.get("chapter_name")
        )

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chapter_number": meta.get("chapter_number"),
                "chapter_name": meta.get("chapter_name")
            })

    print("\n================ LLM RESPONSE ================")
    print(answer_text)
    print("=============================================\n")

    return {
        "answer": answer_text,
        "sources": sources
    }