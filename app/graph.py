"""Question answering over one contract as a LangGraph: retrieve -> generate (Phase 2+ adds nodes)."""
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.llm import get_chat_model
from app.retrieval import ContractRetriever

SYSTEM_PROMPT = (
    "You are a contract review assistant. Answer using ONLY the numbered passages from the contract. "
    "Quote the key words of the clause and cite the passage like [1]. "
    "If the passages do not address the question, say: The contract does not address this. Do not guess."
)
PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "Passages:\n{context}\n\nQuestion: {question}")]
)


class QAState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] (characters {d.metadata['char_start']}-{d.metadata['char_end']})\n{d.page_content}"
                       for i, d in enumerate(docs, 1))


def build_graph(doc_name: str, k: int = 5, chunker: str = "clause", retriever=None, llm=None):
    retriever = retriever or ContractRetriever(doc_name=doc_name, k=k, chunker=chunker)
    chain = PROMPT | (llm or get_chat_model()) | StrOutputParser()

    def retrieve(state: QAState) -> dict:
        return {"documents": retriever.invoke(state["question"])}

    def generate(state: QAState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {"answer": "No contract found with that name."}
        return {"answer": chain.invoke({"context": format_docs(docs), "question": state["question"]})}

    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
