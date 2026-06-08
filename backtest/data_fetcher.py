"""Downloads Bybit V5 linear kline data with full pagination and CSV cache."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

API_BASE_URL = "https://api.bybit.com"
CACHE_DIR = Path("data/cache")
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
REQUEST_SLEEP_SECONDS = 0.5
RATE_LIMIT_SLEEP_SECONDS = 3.0
MAX_RATE_LIMIT_RETRIES = 5
CACHE_END_GRACE = pd.Timedelta(hours=24)


class BybitDataError(RuntimeError):
    """Raised when Bybit data cannot be fetched."""


class CacheDataError(RuntimeError):
    """Raised when cache-only mode cannot satisfy a data request."""


def set_data_source(api_base_url: str | None = None, cache_dir: str | Path | None = None) -> None:
    """Configure Bybit base URL and local cache directory."""
    global API_BASE_URL, CACHE_DIR
    if api_base_url:
        API_BASE_URL = api_base_url.rstrip("/")
    if cache_dir:
        CACHE_DIR = Path(cache_dir)


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}.csv"


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    normalized = df[COLUMNS].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = normalized[column].astype("float64")
    return normalized.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _read_cache(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return None
    cached = _normalize_frame(pd.read_csv(path))
    if cached.empty:
        return None
    start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
    cache_end = cached["timestamp"].max()
    cache_start = cached["timestamp"].min()
    if cache_start - CACHE_END_GRACE <= start_ts and cache_end + CACHE_END_GRACE >= end_ts:
        effective_start = max(start_ts, cache_start)
        effective_end = min(end_ts, cache_end)
        mask = (cached["timestamp"] >= effective_start) & (cached["timestamp"] <= effective_end)
        return cached.loc[mask].reset_index(drop=True)
    return None


def import_klines_csv(input_path: str | Path, symbol: str, interval: str) -> pd.DataFrame:
    """Normalize an external candle CSV into the local cache format."""
    source = Path(input_path)
    df = pd.read_csv(source)
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")
    normalized = _normalize_frame(df)
    if normalized.empty:
        raise ValueError(f"{source} did not contain any usable candle rows")
    _write_cache(symbol, interval, normalized)
    return normalized


def _write_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    path = _cache_path(symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def check_data_source(symbol: str = "BTCUSDT", interval: str = "15", category: str = "linear") -> bool:
    """Return True if the configured Bybit endpoint responds to a tiny kline request."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 60 * 60 * 1000
    response = requests.get(
        urljoin(API_BASE_URL + "/", "v5/market/kline"),
        params={
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": 1,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("retCode") == 0 and bool(payload.get("result", {}).get("list", []))


def fetch_recent_klines(
    symbol: str,
    interval: str,
    limit: int = 100,
    category: str = "linear",
) -> pd.DataFrame:
    """Fetch the most recent Bybit klines directly, bypassing historical cache."""
    payload = None
    last_request_error: requests.RequestException | None = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            response = requests.get(
                urljoin(API_BASE_URL + "/", "v5/market/kline"),
                params={
                    "category": category,
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                },
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            last_request_error = exc
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise BybitDataError(
                    f"Could not fetch recent Bybit klines for {symbol} {interval} after retries"
                ) from exc
            time.sleep(RATE_LIMIT_SLEEP_SECONDS * (attempt + 1))
            continue
        payload = response.json()
        if payload.get("retCode") != 10006:
            break
        if attempt == MAX_RATE_LIMIT_RETRIES:
            raise BybitDataError(f"Bybit rate limit persisted for recent {symbol} {interval}: {payload}")
        time.sleep(RATE_LIMIT_SLEEP_SECONDS * (attempt + 1))
    if payload is None and last_request_error is not None:
        raise BybitDataError(f"Could not fetch recent Bybit klines for {symbol} {interval} after retries") from last_request_error
    if payload.get("retCode") != 0:
        raise BybitDataError(f"Bybit API error for recent {symbol} {interval}: {payload}")
    rows = payload.get("result", {}).get("list", [])
    parsed = [
        {
            "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in rows
    ]
    return _normalize_frame(pd.DataFrame(parsed, columns=COLUMNS))


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    category: str = "linear",
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Return Bybit kline DataFrame with timestamp, open, high, low, close, volume.

    Bybit returns candles newest-first. This function paginates backward and returns
    oldest-first data.
    """
    cached = _read_cache(symbol, interval, start_ms, end_ms)
    if cached is not None:
        return cached
    if cache_only:
        path = _cache_path(symbol, interval)
        raise CacheDataError(
            f"Cache-only mode needs {path} to cover the requested period. "
            "Expected columns: timestamp, open, high, low, close, volume."
        )

    all_rows: list[list[str]] = []
    current_end = end_ms
    while current_end >= start_ms:
        payload = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = requests.get(
                    urljoin(API_BASE_URL + "/", "v5/market/kline"),
                    params={
                        "category": category,
                        "symbol": symbol,
                        "interval": interval,
                        "start": start_ms,
                        "end": current_end,
                        "limit": 1000,
                    },
                    timeout=20,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                path = _cache_path(symbol, interval)
                raise BybitDataError(
                    f"Could not fetch Bybit klines for {symbol} interval {interval} from {API_BASE_URL}. "
                    f"If this is a DNS/network block, run from another network or place a CSV cache at {path} "
                    "with columns: timestamp, open, high, low, close, volume."
                ) from exc
            payload = response.json()
            if payload.get("retCode") != 10006:
                break
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise BybitDataError(f"Bybit rate limit persisted for {symbol} {interval}: {payload}")
            time.sleep(RATE_LIMIT_SLEEP_SECONDS * (attempt + 1))
        if payload is None or payload.get("retCode") != 0:
            raise BybitDataError(f"Bybit API error for {symbol} {interval}: {payload}")
        rows = payload.get("result", {}).get("list", [])
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ms = min(int(row[0]) for row in rows)
        next_end = oldest_ms - 1
        if next_end >= current_end:
            break
        current_end = next_end
        time.sleep(REQUEST_SLEEP_SECONDS)

    parsed = [
        {
            "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in all_rows
        if start_ms <= int(row[0]) <= end_ms
    ]
    df = _normalize_frame(pd.DataFrame(parsed, columns=COLUMNS))
    _write_cache(symbol, interval, df)
    return df


def fetch_all_symbols(
    symbols: list[str],
    intervals: list[str],
    lookback_days: int = 180,
    cache_only: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return nested dict data[symbol][interval] = DataFrame."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        data[symbol] = {}
        for interval in intervals:
            data[symbol][interval] = fetch_klines(symbol, interval, start_ms, end_ms, cache_only=cache_only)
    return data
