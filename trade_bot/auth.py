"""RSA-PSS request signing for Kalshi's authenticated endpoints.

Kalshi signs `timestamp_ms + METHOD + path` (path includes the
/trade-api/v2 prefix but never the query string) with RSA-PSS
(SHA-256, MGF1-SHA256, salt length = digest length), then attaches the
result as three headers. Public market-data endpoints don't need any
of this.
"""
from __future__ import annotations

import base64
import time
from functools import lru_cache

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@lru_cache(maxsize=8)
def load_private_key(path: str) -> rsa.RSAPrivateKey:
    with open(path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(f"Key at {path} is not an RSA private key")
    return key


def sign_pss_text(private_key: rsa.RSAPrivateKey, message: str) -> str:
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256().digest_size,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def build_auth_headers(
    key_id: str, private_key: rsa.RSAPrivateKey, method: str, signed_path: str
) -> dict[str, str]:
    """`signed_path` must include the /trade-api/v2 prefix, no query string."""
    timestamp_ms = str(int(time.time() * 1000))
    message = timestamp_ms + method.upper() + signed_path
    signature = sign_pss_text(private_key, message)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }
