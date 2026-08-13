"""JSON API entry point for the native Android app -- read-mostly status/
positions/trades plus bot start/stop. See trade_bot/api.py for the routes
and the auth model (Caddy gates everything, this trusts the edge).

    uv run python run_api.py

Binds 127.0.0.1 only, same pattern as the dashboard (run app.py via
streamlit) -- Caddy is the only thing that ever reaches this from outside.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("trade_bot.api:app", host="127.0.0.1", port=8502)
