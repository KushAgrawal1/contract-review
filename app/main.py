from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.db import get_conn
from app.graph import build_graph
from app.review import review_contract

app = FastAPI(title="Contract Review Assistant")


class AskRequest(BaseModel):
    doc_name: str
    question: str


class Citation(BaseModel):
    n: int
    char_start: int
    char_end: int
    heading: str | None
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class ReviewRequest(BaseModel):
    doc_name: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/contract")
def contract(doc_name: str) -> dict:
    """The full contract text. The character ranges returned by /review are offsets into this text."""
    with get_conn() as conn:
        row = conn.execute("SELECT text FROM contracts WHERE doc_name = %s", (doc_name,)).fetchone()
    if not row:
        raise HTTPException(404, "Unknown contract. See GET /contracts") from None
    return {"doc_name": doc_name, "text": row[0]}


@app.post("/review")
def review(req: ReviewRequest) -> dict:
    """All 40 clause checks for one contract. About 10 seconds and one cent the first time, instant after."""
    if not settings.openai_api_key:
        raise HTTPException(500, "OPENAI_API_KEY is not set (see .env.example)")
    try:
        return review_contract(req.doc_name)
    except KeyError:
        raise HTTPException(404, "Unknown contract. See GET /contracts") from None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/contracts")
def contracts() -> list[str]:
    with get_conn() as conn:
        return [
            r[0]
            for r in conn.execute("SELECT doc_name FROM contracts ORDER BY doc_name").fetchall()
        ]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not settings.openai_api_key:
        raise HTTPException(500, "OPENAI_API_KEY is not set (see .env.example)")
    with get_conn() as conn:
        if not conn.execute(
            "SELECT 1 FROM contracts WHERE doc_name = %s", (req.doc_name,)
        ).fetchone():
            raise HTTPException(404, "Unknown contract. See GET /contracts") from None
    out = build_graph(req.doc_name, k=settings.top_k).invoke({"question": req.question})
    cites = [
        Citation(
            n=i,
            char_start=d.metadata["char_start"],
            char_end=d.metadata["char_end"],
            heading=d.metadata["heading"],
            snippet=d.page_content[:300],
            score=d.metadata["score"],
        )
        for i, d in enumerate(out["documents"], 1)
    ]
    return AskResponse(answer=out["answer"], citations=cites)
