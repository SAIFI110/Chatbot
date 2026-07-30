from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI


import config

from agent.orchestrator import decide_route
from agent.retriever_agent import retrieve_documents
from agent.answer_agent import generate_answer


# -------------------------
# STATE
# -------------------------

class AgentState(TypedDict, total=False):

    question: str
    documents: list
    answer: str
    route: str
    greeted: bool
    sources: list



# -------------------------
# LLM
# -------------------------

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=0
)



# -------------------------
# ORCHESTRATOR NODE
# -------------------------

def orchestrator_node(state):

    route = decide_route(
        state["question"]
    )

    return {
        "route": route
    }



# -------------------------
# RETRIEVER NODE
# -------------------------

def retriever_node(state):

    docs = retrieve_documents(
        state["question"]
    )

    return {
        "documents": docs
    }



# -------------------------
# ANSWER NODE
# -------------------------

def answer_node(state):

    result = generate_answer(
        state["question"],
        state["documents"]
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }



# -------------------------
# GENERAL NODE
# -------------------------

def general_node(state):

    response = llm.invoke(
        f"""
You are Pakistan Law AI Assistant.

You have only two jobs:

1. Handle greetings and casual conversation.
2. Redirect anything unrelated to Pakistan law.

Rules:

- For first greeting like:
  hi, hello, salam
  introduce yourself.

Example:
"Hello! I am Pakistan Law AI Assistant. How can I help you with Pakistan law today?"

- For casual questions like:
  how are you?
  what are you doing?
  reply politely.

Example:
"I am doing well, thank you for asking. How can I help you with Pakistan law today?"

- For ANY question not related to Pakistan law:
  Do NOT answer the question.
  Reply exactly:

"I only provide information related to Pakistan law."

Forbidden topics:
- sports
- celebrities
- countries
- geography
- weather
- science
- general knowledge


User question:
{state["question"]}
"""
    )

    return {
        "answer": response.content,
        "greeted": True
    }

# -------------------------
# CREATE GRAPH
# -------------------------

workflow = StateGraph(AgentState)



workflow.add_node(
    "orchestrator",
    orchestrator_node
)


workflow.add_node(
    "retriever",
    retriever_node
)


workflow.add_node(
    "answer",
    answer_node
)


workflow.add_node(
    "general",
    general_node
)



# Entry

workflow.set_entry_point(
    "orchestrator"
)



# -------------------------
# ROUTER
# -------------------------

def router(state):

    if state["route"] == "LEGAL":
        return "retriever"

    return "general"



workflow.add_conditional_edges(
    "orchestrator",
    router
)



# -------------------------
# EDGES
# -------------------------

workflow.add_edge(
    "retriever",
    "answer"
)


workflow.add_edge(
    "answer",
    END
)


workflow.add_edge(
    "general",
    END
)



# -------------------------
# MEMORY + COMPILE
# -------------------------

memory = MemorySaver()


app = workflow.compile(
    checkpointer=memory


)

