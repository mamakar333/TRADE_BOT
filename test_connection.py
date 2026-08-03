"""Run this first: verifies public market data works, then verifies
RSA-PSS auth against whatever environment KALSHI_USE_DEMO points at.

    uv run python test_connection.py
"""
from __future__ import annotations

import sys

from trade_bot.client import KalshiAPIError, KalshiClient
from trade_bot.config import get_settings
from trade_bot.data import get_portfolio_balance, list_open_markets


def check_public_data(client: KalshiClient) -> bool:
    print(f"Base URL: {client.settings.base_url}")
    print("\n[1/2] Fetching open markets (no auth)...")
    try:
        markets = list_open_markets(client, max_pages=1, page_limit=5).markets
    except KalshiAPIError as e:
        print(f"  FAILED: {e}")
        return False

    if not markets:
        print("  Request succeeded but returned zero markets. That's odd but not "
              "necessarily broken -- check the environment has open markets.")
    else:
        print(f"  OK -- got {len(markets)} markets, e.g.:")
        for m in markets[:5]:
            print(
                f"    {m.ticker:<25} {m.title[:40]:<40} "
                f"YES bid/ask: {m.yes_bid_pct}/{m.yes_ask_pct}  status={m.status}"
            )
    return True


def check_auth(client: KalshiClient) -> bool:
    print("\n[2/2] Verifying RSA-PSS auth against a signed endpoint...")
    settings = client.settings
    if not settings.api_key_id or not settings.private_key_path:
        print("  SKIPPED -- KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set in .env.")
        return True

    try:
        balance = get_portfolio_balance(client)
    except KalshiAPIError as e:
        print(f"  FAILED: {e}")
        print(
            "  If this is a 401/403, double check: the key ID/PEM match the "
            "environment you're pointed at (KALSHI_USE_DEMO), the PEM path is "
            "correct, and your machine's clock is accurate (signatures are "
            "timestamp-based)."
        )
        return False
    except FileNotFoundError as e:
        print(f"  FAILED: private key file not found -- {e}")
        return False

    print(f"  OK -- signed request accepted. Balance response: {balance}")
    return True


def main() -> int:
    settings = get_settings()
    client = KalshiClient(settings)
    try:
        public_ok = check_public_data(client)
        auth_ok = check_auth(client)
    finally:
        client.close()

    print("\n" + "=" * 50)
    if public_ok and auth_ok:
        print("All checks passed.")
        return 0
    print("Some checks failed -- see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
