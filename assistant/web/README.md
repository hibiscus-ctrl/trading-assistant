# Dashboard

A local web page for running TradingAgents analyses and browsing past runs. It only analyses; it never places orders.

## Run it

```bash
cd ~/projects/trading-assistant          # .env with your model API key lives here
uv pip install -p .venv -e ".[web]"      # once
.venv/bin/tradingagents-web              # http://127.0.0.1:8765, or --port N
```

Open it from VS Code's **Ports** tab (forward port 8765). The server listens on 127.0.0.1 only and has no login, so do not expose the port publicly.

It also refuses any request whose Host is not `localhost`, `127.0.0.1` or `[::1]`, which blocks DNS-rebinding attacks from other websites. If you open it through a proxy that keeps its own hostname (some browser-based VS Code setups do), the page says which host it saw; add it with `TA_WEB_ALLOWED_HOSTS=that.host` (comma-separated) and restart.

## What it does

- **Home:** start an analysis (ticker, date, debate rounds, provider, quick and deep model, analysts) and see every run with its status, rating and token use. Providers appear when their API key is set in `.env`.
- **Run page:** live pipeline progress, the report being written and the agents' messages and tool calls, refreshed every 2 seconds. When the run finishes, the page reloads to the full report with one section per team, plus a link to TradingAgents' own HTML report.
- **Ticker page:** every run for one ticker, newest analysis date first.

Runs go through one worker, one at a time, so a second request waits rather than doubling model spend. **Stop** takes effect at the next step; the model call already in flight is still billed.

## Where things are kept

- Run list: `~/.tradingagents/web/dashboard.db` (SQLite) and a JSON snapshot per run in `~/.tradingagents/web/runs/`.
- Reports and the memory log: where TradingAgents always writes them, under `~/.tradingagents/logs/`.

A run in progress when the server stops is marked failed on the next start.

## Code

All new code is in `assistant/web/`; `tradingagents/` and `cli/` are used as libraries and left unchanged. Tests: `tests/test_web_dashboard.py`.
