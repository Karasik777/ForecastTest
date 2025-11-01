"""
Shared Binance API utilities.

Centralising HTTP calls and timestamp handling keeps the project scripts
consistent and reduces duplicated error-prone code. These helpers keep
dependencies light (requests, pandas, numpy) and return tidy DataFrames
with unified column naming.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Optional

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
_KLINES_ENDPOINT = "/api/v3/klines"
_USER_AGENT = "forecast-test/1.0"
_DEFAULT_TIMEOUT = 30

_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qav",
    "num_trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def ensure_utc(ts_like) -> pd.Timestamp:
    """Return a timezone-aware UTC pandas Timestamp."""
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def iso_to_ms(value) -> int:
    """Convert ISO8601 or Timestamp-like value to epoch milliseconds."""
    return int(ensure_utc(value).timestamp() * 1000)


def _request_klines(params: dict, session: Optional[requests.Session] = None) -> list:
    """Low level REST call with basic error handling."""
    sess = session or requests.Session()
    headers = {"User-Agent": _USER_AGENT}
    resp = sess.get(
        BINANCE_BASE + _KLINES_ENDPOINT,
        params=params,
        headers=headers,
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _klines_to_frame(raw: Iterable) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=_COLUMNS)
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_recent_klines(
    symbol: str,
    interval: str,
    limit: int,
    *,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch the most recent `limit` klines for the given symbol/interval.

    Parameters mirror Binance REST API v3. The result is a pandas DataFrame
    with UTC timestamps and numeric OHLCV columns.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    raw = _request_klines(params, session=session)
    return _klines_to_frame(raw)


def fetch_range_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int = 1000,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch klines between the provided epoch millisecond bounds.

    Binance caps each response at 1000 rows; callers that need more must
    chunk manually.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": limit,
    }
    raw = _request_klines(params, session=session)
    return _klines_to_frame(raw)


@contextlib.contextmanager
def session_scope(session: Optional[requests.Session] = None):
    """
    Context manager that yields a requests.Session re-using the caller's
    instance if supplied, otherwise creating (and closing) a temporary one.
    """
    own = session is None
    sess = session or requests.Session()
    try:
        yield sess
    finally:
        if own:
            sess.close()

