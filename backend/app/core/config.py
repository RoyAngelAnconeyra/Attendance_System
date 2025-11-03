from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = Field(default="attendance-face")
    ENV: str = Field(default="dev")
    API_PREFIX: str = Field(default="/api")

    DB_HOST: str = Field(default="127.0.0.1")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="attendance_db")
    DB_USER: str = Field(default="root")
    DB_PASSWORD: str = Field(default="root")

    JWT_SECRET: str = Field(default="CHANGE_ME_SUPER_SECRET")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=120)

    CAMERA_INDEX: int = Field(default=0)
    CONFIDENCE_THRESHOLD: float = Field(default=0.6)

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        # Load env from backend/.env regardless of where uvicorn is started
        env_file = str(Path(__file__).resolve().parents[2] / ".env")

settings = Settings()
