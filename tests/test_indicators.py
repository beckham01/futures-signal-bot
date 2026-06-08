import pandas as pd

from backtest.indicators import atr, ema, ema_slope, rsi, volume_sma


def test_ema_simple_series():
    data = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = ema(data, period=5)
    assert abs(result.iloc[-1] - 14.1756) < 0.01


def test_rsi_overbought():
    data = pd.Series([float(value) for value in range(1, 30)])
    result = rsi(data, period=14)
    assert result.iloc[-1] > 99


def test_rsi_oversold():
    data = pd.Series([float(value) for value in range(30, 1, -1)])
    result = rsi(data, period=14)
    assert result.iloc[-1] < 1


def test_atr_known_values():
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 13.0, 14.0],
            "low": [8.0, 9.0, 10.0, 12.0],
            "close": [9.0, 11.0, 12.0, 13.0],
        }
    )
    result = atr(df, period=3)
    assert abs(result.iloc[-1] - 2.3704) < 0.01


def test_volume_sma():
    result = volume_sma(pd.Series([1, 2, 3, 4, 5]), period=3)
    assert result.iloc[-1] == 4.0


def test_ema_slope_direction():
    series = pd.Series([10.0, 11.0, 12.0, 13.0])
    result = ema_slope(series, lookback=3)
    assert result.iloc[-1] > 0
