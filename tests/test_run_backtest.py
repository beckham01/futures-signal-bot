from backtest.run_backtest import parse_args


def test_parse_args_accepts_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run_backtest", "--config", "candidate_config.yaml", "--symbols", "BTCUSDT"],
    )
    args = parse_args()
    assert args.config == "candidate_config.yaml"
    assert args.symbols == ["BTCUSDT"]
