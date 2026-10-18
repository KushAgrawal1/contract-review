"""Second-stage reranker: re-order the candidates from retrieval, best first.

A cross-encoder reads the question and one passage TOGETHER and scores how well the passage answers it. That is
slower than comparing embeddings but much more precise, so it is only run on the top ~20 candidates.
Needs: pip install -r requirements-rerank.txt (first run downloads about 1 GB of model weights).
"""
from functools import lru_cache

from langchain_core.documents import Document

from app.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if len(docs) <= 1:
            return docs
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("pip install -r requirements-rerank.txt to use --rerank bge") from exc
            self._model = CrossEncoder(self.model_name, max_length=512)
        scores = self._model.predict([(query, d.page_content) for d in docs], batch_size=16)
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
        out = []
        for i in order:
            d = docs[i].model_copy(deep=True)
            d.metadata["rerank_score"] = float(scores[i])
            out.append(d)
        return out


@lru_cache(maxsize=2)
def get_reranker(kind: str):
    if kind == "bge":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker: {kind}")
