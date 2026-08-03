"""Environment-driven configuration for the Kalshi client."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROD_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_DEMO_BASE_URL = "https://external-api.demo.kalshi.co/trade-api/v2"

# Signature strings always use the API path prefix, regardless of which
# host (prod/demo) a request is actually sent to.
API_PATH_PREFIX = "/trade-api/v2"


@dataclass(frozen=True)
class Settings:
    api_key_id: str | None
    private_key_path: str | None
    prod_base_url: str
    demo_base_url: str
    use_demo: bool

    @property
    def base_url(self) -> str:
        return self.demo_base_url if self.use_demo else self.prod_base_url


def get_settings() -> Settings:
    use_demo = os.getenv("KALSHI_USE_DEMO", "true").strip().lower() in ("1", "true", "yes")
    return Settings(
        api_key_id=os.getenv("KALSHI_API_KEY_ID") or None,
        private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH") or None,
        prod_base_url=os.getenv("KALSHI_BASE_URL", DEFAULT_PROD_BASE_URL),
        demo_base_url=os.getenv("KALSHI_DEMO_BASE_URL", DEFAULT_DEMO_BASE_URL),
        use_demo=use_demo,
    )
