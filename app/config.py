from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 10
    agent_timeout_seconds: int = 30
    storage_path: str = "./storage"

    class Config:
        env_file = ".env"

settings = Settings()