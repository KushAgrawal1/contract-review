from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(model: str | None = None, **kwargs) -> ChatOpenAI:
    # temperature=0 keeps eval runs comparable between phases.
    # Reasoning models reject this argument: delete it if you switch to one.
    return ChatOpenAI(
        model=model or settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_retries=6,
        **kwargs,
    )
