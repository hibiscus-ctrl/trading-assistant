"""SQLite record of dashboard runs: what was asked, how it went, where the report is.

The reports themselves stay where TradingAgents writes them; this table only
indexes them. A JSON snapshot per run (sections, messages, stats) sits beside the
database so a finished run can be shown after a restart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

_COLUMNS = (
    "id", "ticker", "trade_date", "status", "rating", "created_at", "started_at",
    "finished_at", "provider", "quick_model", "deep_model", "analysts", "depth",
    "llm_calls", "tool_calls", "tokens_in", "tokens_out", "report_path", "error",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    trade_date   TEXT NOT NULL,
    status       TEXT NOT NULL,
    rating       TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    provider     TEXT NOT NULL,
    quick_model  TEXT NOT NULL,
    deep_model   TEXT NOT NULL,
    analysts     TEXT NOT NULL,
    depth        INTEGER NOT NULL,
    llm_calls    INTEGER NOT NULL DEFAULT 0,
    tool_calls   INTEGER NOT NULL DEFAULT 0,
    tokens_in    INTEGER NOT NULL DEFAULT 0,
    tokens_out   INTEGER NOT NULL DEFAULT 0,
    report_path  TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS runs_ticker ON runs (ticker, trade_date);
"""

ACTIVE = ("queued", "running")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.snapshots = self.data_dir / "runs"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.data_dir / "dashboard.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            # Nothing survives a restart mid-run: say so rather than show it as live.
            self._db.execute(
                "UPDATE runs SET status = 'failed', error = 'Interrupted: the dashboard restarted', "
                "finished_at = ? WHERE status IN ('queued', 'running')",
                (now_iso(),),
            )
            self._db.commit()

    def create(self, *, ticker: str, trade_date: str, provider: str, quick_model: str,
               deep_model: str, analysts: list[str], depth: int) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO runs (ticker, trade_date, status, created_at, provider, quick_model, "
                "deep_model, analysts, depth) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)",
                (ticker, trade_date, now_iso(), provider, quick_model, deep_model,
                 json.dumps(analysts), depth),
            )
            self._db.commit()
            return int(cursor.lastrowid)

    def update(self, run_id: int, **fields) -> None:
        unknown = set(fields) - set(_COLUMNS)
        if unknown:
            raise ValueError(f"unknown run fields: {sorted(unknown)}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock:
            self._db.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id))
            self._db.commit()

    def get(self, run_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _as_dict(row) if row else None

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_as_dict(row) for row in rows]

    def for_ticker(self, ticker: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM runs WHERE ticker = ? ORDER BY trade_date DESC, id DESC", (ticker,)
            ).fetchall()
        return [_as_dict(row) for row in rows]

    def save_snapshot(self, run_id: int, snapshot: dict) -> None:
        path = self.snapshots / f"{run_id}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(snapshot), encoding="utf-8")
        temp.replace(path)

    def load_snapshot(self, run_id: int) -> dict | None:
        try:
            return json.loads((self.snapshots / f"{run_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


def _as_dict(row: sqlite3.Row) -> dict:
    record = dict(row)
    record["analysts"] = json.loads(record["analysts"])
    return record
