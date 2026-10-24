from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    openai_api_key: str = ""
    embedding_model: str = (
        "text-embedding-3-small"  # 1536 dims, matches vector(1536) in sql/init.sql
    )
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"


settings = Settings()
