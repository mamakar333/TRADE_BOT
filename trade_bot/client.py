"""Thin HTTP client for the Kalshi REST API.

Public market-data endpoints (markets, events, series, orderbooks) are
sent unsigned. Portfolio/order endpoints require RSA-PSS signed
headers via `signed=True`.

Order placement (`create_order`) uses `POST /portfolio/events/orders`, the
current V2 order endpoint -- confirmed 2026-08-01 against Kalshi's live
openapi.yaml that the older `/portfolio/orders` create-order path (action/
side/yes_price/no_price) no longer exists in the spec at all (only GET
remains there); V2 is the only working order-creation endpoint as of this
writing. V2 quotes a single unified order book in YES-scale price: side
"bid" buys YES, side "ask" sells YES (economically identical to buying NO
at `1 - price`), per the BookSide schema. See `create_order`'s docstring for
the exact side/price mapping used to express buy-YES/buy-NO/close-YES/
close-NO in those terms.
"""
from __future__ import annotations

import random
import time
import uuid
from typing import Any

import httpx

from .auth import build_auth_headers, load_private_key
from .config import API_PATH_PREFIX, Settings, get_settings

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class KalshiAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class KalshiClient:
    def __init__(self, settings: Settings | None = None, timeout: float = 10.0):
        self.settings = settings or get_settings()
        self._http = httpx.Client(base_url=self.settings.base_url, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _signed_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.settings.api_key_id:
            raise KalshiAPIError("KALSHI_API_KEY_ID is not set; cannot sign request")
        if not self.settings.private_key_path:
            raise KalshiAPIError("KALSHI_PRIVATE_KEY_PATH is not set; cannot sign request")
        private_key = load_private_key(self.settings.private_key_path)
        signed_path = API_PATH_PREFIX + path
        return build_auth_headers(self.settings.api_key_id, private_key, method, signed_path)

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> dict[str, Any]:
        headers = self._signed_headers("GET", path) if signed else {}

        attempt = 0
        while True:
            try:
                response = self._http.get(path, params=params, headers=headers)
            except httpx.TransportError as e:
                # Covers ConnectError, ReadTimeout, RemoteProtocolError, etc. -- a
                # dropped keep-alive connection or blip mid-request never reaches
                # the status-code check below, so it needs its own retry path.
                # Without this, a single transient network hiccup crashes an
                # otherwise-long-running 24/7 process (observed in production
                # 2026-07-12: a RemoteProtocolError killed the paper-trading loop).
                attempt += 1
                if attempt >= max_retries:
                    raise KalshiAPIError(f"GET {path} failed after {max_retries} attempts: {e}") from e
                sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(sleep_for)
                if signed:
                    headers = self._signed_headers("GET", path)
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES:
                break
            attempt += 1
            if attempt >= max_retries:
                break
            sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            if signed:
                # Timestamp must stay fresh across retries.
                headers = self._signed_headers("GET", path)

        if response.status_code >= 400:
            raise KalshiAPIError(
                f"GET {path} failed with {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return response.json()

    def post(self, path: str, json_body: dict[str, Any], signed: bool = True) -> dict[str, Any]:
        """Single attempt, deliberately no retry loop -- unlike `get()`. A GET
        can be retried freely because it never changes state; retrying a POST
        that places a real order risks submitting it twice if the first
        attempt actually succeeded but the response was lost. If this raises,
        the caller must NOT assume the order didn't happen -- treat it as
        unknown and reconcile against GET /portfolio/positions before acting
        again, never blindly resubmit."""
        headers = self._signed_headers("POST", path) if signed else {}
        headers["Content-Type"] = "application/json"
        try:
            response = self._http.post(path, json=json_body, headers=headers)
        except httpx.TransportError as e:
            raise KalshiAPIError(f"POST {path} failed (state unknown, do not blindly retry): {e}") from e

        if response.status_code >= 400:
            raise KalshiAPIError(
                f"POST {path} failed with {response.status_code}: {response.text}",
                status_code=response.status_code,
            )
        return response.json()

    def create_order(
        self,
        ticker: str,
        book_side: str,
        price_pct: float,
        count: int,
        time_in_force: str = "fill_or_kill",
        self_trade_prevention_type: str = "taker_at_cross",
        client_order_id: str | None = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Places a real order via `POST /portfolio/events/orders`. `book_side`
        is "bid" (buy YES) or "ask" (sell YES / buy NO at `1 - price`) --
        Kalshi's V2 endpoint has one unified, YES-scale order book, not
        separate yes/no books. `price_pct` is 0-100 on that YES scale. To
        express the four trading actions using the yes_bid_pct/yes_ask_pct
        already on a MarketSnapshot:

            buy YES (open long)   -> book_side="bid", price_pct=yes_ask_pct
            buy NO  (open short)  -> book_side="ask", price_pct=yes_bid_pct
            close a YES position  -> book_side="ask", price_pct=yes_bid_pct
            close a NO position   -> book_side="bid", price_pct=yes_ask_pct

        Defaults to fill_or_kill (fills completely immediately or not at all)
        so this never leaves a resting order behind that fills later at a
        price nobody evaluated, and to taker_at_cross self-trade prevention.
        `client_order_id` defaults to a fresh UUID (Kalshi dedupes on it) --
        pass one in explicitly only when deliberately retrying a known-failed
        attempt, never generate a new one for what might be the same order.
        """
        if book_side not in ("bid", "ask"):
            raise ValueError(f"book_side must be 'bid' or 'ask', got {book_side!r}")
        if not (0 < price_pct < 100):
            raise ValueError(f"price_pct must be between 0 and 100 exclusive, got {price_pct}")
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        body = {
            "ticker": ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side": book_side,
            "count": f"{count}.00",
            "price": f"{price_pct / 100:.2f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
        }
        if reduce_only:
            body["reduce_only"] = True
        return self.post("/portfolio/events/orders", body, signed=True)
