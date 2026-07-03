import pandas as pd

from backtest.resample import resample_ohlcv


def test_resample_15m_to_30m_ohlcv():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min"),
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = resample_ohlcv(df, 30)

    assert len(result) == 2
    assert result.iloc[0]["open"] == 1.0
    assert result.iloc[0]["high"] == 3.0
    assert result.iloc[0]["low"] == 0.5
    assert result.iloc[0]["close"] == 2.5
    assert result.iloc[0]["volume"] == 30.0
