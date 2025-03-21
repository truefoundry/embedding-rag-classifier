from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings class to hold all the environment variables
    """

    LOG_LEVEL: str = "INFO"

    HF_TOKEN: str

    LLM_GATEWAY_URL: str
    LLM_GATEWAY_API_KEY: str
    LLM_MODEL_NAME: str

    INFINITY_URL: str
    INFINITY_API_KEY: str

    QDRANT_URL: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


settings = Settings()
