from pydantic import BaseSettings


class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://mongodb:27017"
    MONGO_DB: str = "campus_kb"
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTORSTORE_PATH: str = "vector_store/faiss_index"

    class Config:
        env_file = ".env"


settings = Settings()
