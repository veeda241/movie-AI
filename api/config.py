from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = "sqlite:///./storage/movie_flow.db"
    storage_root: str = "./storage"
    jwt_secret: str = "change-me-to-a-long-random-string"
    jwt_expire_minutes: int = 10080
    cors_origins: str = "http://localhost:3000"
    frontend_url: str = "http://localhost:3000"

    credit_cost_image: int = 1
    credit_cost_video: int = 5
    credit_cost_movie_scene: int = 8
    credit_cost_assemble: int = 1
    starter_monthly_credits: int = 500
    pro_monthly_credits: int = 2000
    free_signup_credits: int = 50

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_starter: str = ""
    stripe_price_pro: str = ""
    stripe_price_credits_100: str = ""
    stripe_price_credits_500: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
