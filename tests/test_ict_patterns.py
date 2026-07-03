import pandas as pd

from backtest.ict_patterns import (
    detect_breaker_block,
    detect_fvgs,
    find_confluence_zone,
    is_price_in_zone,
    mark_filled_fvgs,
    overlap_zone,
    zones_overlap,
)


def frame(rows):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC"),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [100.0] * len(rows),
        }
    )


def test_bullish_fvg_detected():
    df = frame([(99, 100, 98, 99), (100, 103, 99, 102), (102, 104, 101, 103), (103, 104, 102, 103)])
    fvgs = detect_fvgs(df, 3, pd.Series([1.0] * 4))
    assert fvgs[0]["type"] == "bullish"
    assert fvgs[0]["zone"] == (100.0, 101.0)


def test_bearish_fvg_detected():
    df = frame([(103, 104, 102, 103), (103, 104, 99, 100), (99, 100, 98, 99), (99, 100, 98, 99)])
    fvgs = detect_fvgs(df, 3, pd.Series([1.0] * 4))
    assert fvgs[0]["type"] == "bearish"
    assert fvgs[0]["zone"] == (100.0, 102.0)


def test_fvg_below_min_size_rejected():
    df = frame([(99, 100, 98, 99), (100, 103, 99, 102), (102, 104, 100.1, 103), (103, 104, 102, 103)])
    assert detect_fvgs(df, 3, pd.Series([1.0] * 4), min_gap_atr_multiplier=0.2) == []


def test_fvg_expired_after_48_candles():
    rows = [(99, 100, 98, 99), (100, 103, 99, 102), (102, 104, 101, 103)] + [(103, 104, 102, 103)] * 50
    assert detect_fvgs(frame(rows), 51, pd.Series([1.0] * len(rows)), max_age_candles=48) == []


def test_fvg_filled_when_price_closes_through():
    df = frame(
        [
            (99, 100, 98, 99),
            (100, 103, 99, 102),
            (102, 104, 101, 103),
            (103, 104, 99, 99.5),
        ]
    )
    assert detect_fvgs(df, 3, pd.Series([1.0] * 4)) == []
    marked = mark_filled_fvgs(
        [{"type": "bullish", "zone": (100.0, 101.0), "formed_at": 1, "filled": False}],
        df,
        3,
    )
    assert marked[0]["filled"] is True


def test_fvg_no_lookahead():
    df = frame([(99, 100, 98, 99), (100, 103, 99, 102), (102, 104, 101, 103), (103, 104, 102, 103)])
    atr = pd.Series([1.0] * 4)
    assert detect_fvgs(df, 2, atr) == []
    assert len(detect_fvgs(df, 3, atr)) == 1


def breaker_frame(direction="bullish"):
    rows = []
    for i in range(16):
        rows.append((105 + i * 0.1, 106 + i * 0.1, 104 + i * 0.1, 105 + i * 0.1))
    if direction == "bullish":
        rows[5] = (103, 104, 102, 102.5)
        rows[6] = (102, 103, 99, 101)
        rows[7] = (101, 105, 100, 104)
        rows[8] = (104, 106, 101, 105)
        rows[9] = (105, 107, 102, 106)
    else:
        rows[5] = (102, 104, 101, 103.5)
        rows[6] = (104, 108, 103, 105)
        rows[7] = (105, 106, 101, 102)
        rows[8] = (102, 103, 99, 100)
        rows[9] = (100, 101, 98, 99)
        rows[10] = (99, 100, 98, 99)
    return frame(rows)


def test_bullish_breaker_detected():
    breaker = detect_breaker_block(breaker_frame("bullish"), 10, pd.Series([1.0] * 16), "bullish")
    assert breaker["zone"] == (102.0, 104.0)


def test_bearish_breaker_detected():
    breaker = detect_breaker_block(breaker_frame("bearish"), 10, pd.Series([1.0] * 16), "bearish")
    assert breaker["zone"] == (101.0, 104.0)


def test_breaker_not_confirmed_too_early():
    assert detect_breaker_block(breaker_frame("bullish"), 8, pd.Series([1.0] * 16), "bullish") is None


def test_breaker_confirmed_after_3_candles():
    assert detect_breaker_block(breaker_frame("bullish"), 9, pd.Series([1.0] * 16), "bullish") is not None


def test_breaker_expired_after_96_candles():
    df = pd.concat([breaker_frame("bullish"), frame([(106, 107, 105, 106)] * 100)], ignore_index=True)
    assert detect_breaker_block(df, 110, pd.Series([1.0] * len(df)), "bullish", max_age_candles=96) is None


def test_breaker_mitigated_after_close_through():
    df = breaker_frame("bullish")
    df.loc[10, "close"] = 101.5
    assert detect_breaker_block(df, 10, pd.Series([1.0] * 16), "bullish") is None


def test_zone_helpers():
    assert zones_overlap((100, 110), (105, 115)) is True
    assert zones_overlap((100, 105), (110, 115)) is False
    assert overlap_zone((100, 110), (105, 115)) == (105, 110)
    assert is_price_in_zone(107, (105, 110)) is True


def test_confluence_zone_correct_range():
    breaker = {"zone": (100, 110)}
    fvgs = [{"type": "bullish", "zone": (105, 115)}]
    assert find_confluence_zone(breaker, fvgs, "bullish") == (105, 110)
    assert find_confluence_zone(breaker, fvgs, "bearish") is None
