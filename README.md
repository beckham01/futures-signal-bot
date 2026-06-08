# Futures Signal Bot

Phase 1 implements a Python backtesting engine for a Bybit V5 crypto futures signal strategy.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run Tests

```bash
pytest tests/
```

## Run Backtest

```bash
python -m backtest.run_backtest
python -m backtest.run_backtest --symbols BTCUSDT ETHUSDT --days 90
python -m backtest.run_backtest --config candidate_config.yaml
```

The CLI writes `backtest_report.txt` and exits with code `1` if total R is negative, win rate is below 40%, or fewer than 30 trades are evaluated.

## Check Bybit Access

```bash
python -m backtest.run_backtest --check-data-source
```

The default endpoint is `https://api.bytick.com`, Bybit's alternate mainnet API host. If needed, override it:

```bash
python -m backtest.run_backtest --check-data-source --api-base-url https://api.bybit.com
```

If the check fails with DNS or connection errors, the code is not reaching Bybit from the current network. Use one of these routes:

- Run the same command from another network, VPS, or cloud shell.
- Override the endpoint if Bybit gives you another working regional/API base URL:

```bash
python -m backtest.run_backtest --api-base-url https://api.bytick.com --symbols BTCUSDT ETHUSDT --days 90
```

- Put historical candle CSVs directly into `data/cache/`.

Cache files must be named `{SYMBOL}_{INTERVAL}.csv`, for example:

```text
data/cache/BTCUSDT_15.csv
data/cache/BTCUSDT_60.csv
```

Each CSV must contain:

```text
timestamp,open,high,low,close,volume
```

`timestamp` can be an ISO datetime or an epoch timestamp parseable by pandas.

You can import external OHLCV CSV files into the cache:

```bash
python -m backtest.import_csv --symbol BTCUSDT --interval 15 --input path/to/BTCUSDT_15.csv
python -m backtest.import_csv --symbol BTCUSDT --interval 60 --input path/to/BTCUSDT_60.csv
```

Then run without any network calls:

```bash
python -m backtest.run_backtest --symbols BTCUSDT --days 90 --cache-only
```

## Report Interpretation

`TP2_HIT` is a full winner, `TP1_ONLY` means the first target was hit and the remainder stopped at breakeven, `STOP_HIT` is a full loss, and `OPEN` means the trade was unresolved after 96 bars.

The report also breaks results down by symbol, direction, month, confidence label, confidence score bucket, and each true confidence condition. Use those sections to judge whether the confidence score is actually predictive before raising `min_confidence`.

## Optimize Strategy

Run walk-forward optimization before changing live signal behavior:

```bash
python -m backtest.optimize --days 180
```

The optimizer tests symbol universes, volume thresholds, pullback tolerance, cooldown, ATR filters, and confidence thresholds. It writes:

```text
optimize_results.csv
recommended_config.yaml
```

`recommended_config.yaml` is only written when a candidate passes validation. It does not replace `config.yaml` automatically.

For a faster smoke test:

```bash
python -m backtest.optimize --days 180 --max-runs 20
```

Analyze optimizer outputs before adopting any candidate:

```bash
python -m backtest.analyze_optimization optimize_core_liquid.csv optimize_quarantine_weak.csv
```

The analyzer writes:

```text
optimization_analysis.csv
optimization_analysis_report.txt
candidate_config.yaml
```

`candidate_config.yaml` is review-only. Rerun a normal backtest with those settings before replacing `config.yaml`.

```bash
python -m backtest.run_backtest --config candidate_config.yaml
```

## Run Live Bot

Set Telegram secrets in `.env` at the workspace root or project root:

```bash
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

Run a bounded preflight before starting the continuous bot:

```bash
python -m bot.preflight
python -m bot.preflight --send-test-message
```

The preflight verifies `.env`, Bybit access, and one immediate scan. `--send-test-message` also sends a Telegram confirmation message.

Start the continuous bot:

```bash
python -m bot.main
```

The bot scans every 15-minute candle close, fetches fresh Bybit candles from `https://api.bytick.com`, and sends only signals generated on the latest candle.

Telegram commands:

```text
/menu
/help
/id
/status
/watchlist
/lastsignal
```

## Private Telegram Access

`TELEGRAM_CHAT_ID` is bootstrapped as the permanent owner/admin. Command polling must be enabled for access management.

User access:

```text
/start CODE
/menu
/help
/id
/status
/watchlist
/lastsignal
```

Admin commands:

```text
/code_create
/code_list
/code_revoke CODE
/users
/kick USER_ID
/admin_list
```

Owner-only commands:

```text
/admin_add USER_ID
/admin_remove USER_ID
```

Access codes are one-time use and expire after 24 hours. Active admins and active authorized users receive all signal alerts.

## Deploy To Fly.io

Create the app and persistent state volume:

```bash
fly apps create futures-signal-bot
fly volumes create bot_state --size 1 --region sin
fly secrets set TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=xxx
fly deploy
```

State is stored at `state/bot_state.json` so cooldowns survive restarts.

## Disclaimer

Signals are for research only and are not financial advice.
