import pandas as pd
import pytest

from backtest import data_fetcher


class DummyResponse:
    def __init__(self, rows=None, ret_code=0):
        self.rows = rows or []
        self.ret_code = ret_code

    def raise_for_status(self):
        return None

    def json(self):
        return {"retCode": self.ret_code, "result": {"list": self.rows}}


def test_fetch_klines_orders_and_caches(monkeypatch, tmp_path):
    data_fetcher.set_data_source("https://example.test", tmp_path)
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return DummyResponse(
            [
                ["1700000900000", "11", "12", "10", "11.5", "100", "0"],
                ["1700000000000", "10", "11", "9", "10.5", "90", "0"],
            ]
        )

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    df = data_fetcher.fetch_klines("BTCUSDT", "15", 1700000000000, 1700000900000)

    assert list(df["close"]) == [10.5, 11.5]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert (tmp_path / "BTCUSDT_15.csv").exists()
    assert len(calls) == 1
    assert calls[0][0][0] == "https://example.test/v5/market/kline"

    cached = data_fetcher.fetch_klines("BTCUSDT", "15", 1700000000000, 1700000900000)
    assert len(calls) == 1
    assert list(cached["open"]) == [10.0, 11.0]


def test_cache_only_raises_when_cache_missing(tmp_path):
    data_fetcher.set_data_source("https://example.test", tmp_path)
    with pytest.raises(data_fetcher.CacheDataError):
        data_fetcher.fetch_klines("ETHUSDT", "60", 1700000000000, 1700000900000, cache_only=True)


def test_import_klines_csv_normalizes_to_cache(tmp_path):
    data_fetcher.set_data_source("https://example.test", tmp_path / "cache")
    source = tmp_path / "raw.csv"
    source.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2023-11-14T22:28:20Z,11,12,10,11.5,100\n"
        "2023-11-14T22:13:20Z,10,11,9,10.5,90\n",
        encoding="utf-8",
    )

    imported = data_fetcher.import_klines_csv(source, "BTCUSDT", "15")

    assert list(imported["close"]) == [10.5, 11.5]
    assert (tmp_path / "cache" / "BTCUSDT_15.csv").exists()


def test_fetch_recent_klines_bypasses_cache(monkeypatch, tmp_path):
    data_fetcher.set_data_source("https://example.test", tmp_path)
    (tmp_path / "BTCUSDT_15.csv").write_text(
        "timestamp,open,high,low,close,volume\n2023-11-14T22:13:20Z,10,11,9,10.5,90\n",
        encoding="utf-8",
    )

    def fake_get(*args, **kwargs):
        return DummyResponse([["1700000900000", "11", "12", "10", "11.5", "100", "0"]])

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    df = data_fetcher.fetch_recent_klines("BTCUSDT", "15", limit=1)
    assert list(df["close"]) == [11.5]


def test_fetch_recent_klines_retries_rate_limit(monkeypatch, tmp_path):
    data_fetcher.set_data_source("https://example.test", tmp_path)
    responses = [
        DummyResponse(ret_code=10006),
        DummyResponse([["1700000900000", "11", "12", "10", "11.5", "100", "0"]]),
    ]
    sleeps = []

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    monkeypatch.setattr(data_fetcher.time, "sleep", lambda seconds: sleeps.append(seconds))

    df = data_fetcher.fetch_recent_klines("BTCUSDT", "15", limit=1)

    assert list(df["close"]) == [11.5]
    assert sleeps == [data_fetcher.RATE_LIMIT_SLEEP_SECONDS]


def test_fetch_recent_klines_retries_request_exception(monkeypatch, tmp_path):
    data_fetcher.set_data_source("https://example.test", tmp_path)
    calls = {"count": 0}
    sleeps = []

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise data_fetcher.requests.ConnectTimeout("temporary timeout")
        return DummyResponse([["1700000900000", "11", "12", "10", "11.5", "100", "0"]])

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    monkeypatch.setattr(data_fetcher.time, "sleep", lambda seconds: sleeps.append(seconds))

    df = data_fetcher.fetch_recent_klines("BTCUSDT", "15", limit=1)

    assert list(df["close"]) == [11.5]
    assert calls["count"] == 2
    assert sleeps == [data_fetcher.RATE_LIMIT_SLEEP_SECONDS]
