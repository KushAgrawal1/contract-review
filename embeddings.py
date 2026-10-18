from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    # chunk_size=200 keeps each API request well under OpenAI's per-request token limit,
    # because numeric tables in filings tokenise into many tokens per character.
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        chunk_size=200,
        max_retries=6,
    )
