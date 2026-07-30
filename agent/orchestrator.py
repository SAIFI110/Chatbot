from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)



def decide_route(question):

    prompt = f"""
You are a router for a Pakistan Law Assistant.

Classify the user's message into one category:

LEGAL:
- Pakistan laws
- Penal Code sections
- crimes
- punishments
- offences
- legal definitions
- laws

GENERAL:
- greetings
- casual conversation
- identity questions
- general chat

Return ONLY one word:
LEGAL or GENERAL

User message:
{question}
"""

    response = llm.invoke(prompt)

    return response.content.strip()
    