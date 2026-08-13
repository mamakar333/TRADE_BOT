"""Continuous BTC/USD price collector -- polls Coinbase's public spot price
every second and stores it in btc_price_history.db, independent of Kalshi
and independent of both trading bots. See trade_bot/btc_price_history.py's
docstring for why an external price source at all.

    uv run python run_btc_price_collector.py

Runs as its own always-on systemd service (deploy/btc-price-collector.service)
-- unlike the trading bots, there's no real-money risk here and nothing to
toggle on/off from the dashboard, so it's simplest to just always run once
deployed, the same way api.service/dashboard.service do.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

import httpx

from trade_bot.btc_price_history import BtcPriceHistory, fetch_btc_price_usd

POLL_INTERVAL_SECONDS = 1.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class _StopRequested(BaseException):
    """Raised from the SIGTERM handler -- same pattern as run_live_trading.py's
    and run_paper_trading.py's, see either for why this subclasses
    BaseException rather than Exception."""


def _handle_sigterm(signum, frame) -> None:
    raise _StopRequested()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)

    history = BtcPriceHistory()
    print(f"Collecting BTC/USD price every {POLL_INTERVAL_SECONDS}s from Coinbase's public spot endpoint.")
    print(f"Writing to {history.db_path}")

    consecutive_failures = 0
    try:
        with httpx.Client(timeout=5.0) as client:
            while True:
                cycle_start = time.monotonic()
                price = fetch_btc_price_usd(client)
                if price is not None:
                    history.record_tick(price)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures % 30 == 1:
                        # Logged every ~30 failed ticks (not every single
                        # one) so a real outage is visible in the log
                        # without spamming it once per second the whole
                        # time Coinbase is down.
                        logger.warning("%d consecutive failed price fetches", consecutive_failures)
                elapsed = time.monotonic() - cycle_start
                time.sleep(max(0.0, POLL_INTERVAL_SECONDS - elapsed))
    except (KeyboardInterrupt, _StopRequested):
        print("\nStopped.")
    finally:
        history.close()


if __name__ == "__main__":
    main()
