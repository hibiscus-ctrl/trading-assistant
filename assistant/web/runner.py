"""Runs analyses one at a time on a background thread, the way ``cli/run.py`` does.

A run takes minutes and spends money on every model call, so there is one worker:
a second request waits in the queue rather than doubling the spend.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant.web.progress import Progress, sections
from assistant.web.store import RunStore, now_iso
from cli.display import classify_message_type
from cli.stats_handler import StatsCallbackHandler
from tradingagents.agents.rating import run_rating
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


@dataclass
class RunRequest:
    ticker: str
    trade_date: str
    asset_type: str
    analysts: list[str]
    provider: str
    backend_url: str | None
    quick_model: str
    deep_model: str
    depth: int


@dataclass
class LiveRun:
    id: int
    request: RunRequest
    progress: Progress
    stats: StatsCallbackHandler = field(default_factory=StatsCallbackHandler)
    status: str = "queued"
    started: float | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def elapsed(self) -> float:
        return time.time() - self.started if self.started else 0.0


def build_config(request: RunRequest) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = request.provider
    config["backend_url"] = request.backend_url
    config["quick_think_llm"] = request.quick_model
    config["deep_think_llm"] = request.deep_model
    config["max_debate_rounds"] = request.depth
    config["max_risk_discuss_rounds"] = request.depth
    # The dashboard never resumes a half-finished run; each request starts fresh.
    config["checkpoint_enabled"] = False
    return config


class RunManager:
    def __init__(self, store: RunStore, graph_factory: Callable[..., TradingAgentsGraph] = TradingAgentsGraph,
                 start_worker: bool = True):
        self.store = store
        self.graph_factory = graph_factory
        self._queue: queue.Queue[LiveRun] = queue.Queue()
        self._live: dict[int, LiveRun] = {}
        self._lock = threading.Lock()
        if start_worker:
            threading.Thread(target=self._work, name="dashboard-runner", daemon=True).start()

    def submit(self, request: RunRequest) -> int:
        run_id = self.store.create(
            ticker=request.ticker, trade_date=request.trade_date, provider=request.provider,
            quick_model=request.quick_model, deep_model=request.deep_model,
            analysts=request.analysts, depth=request.depth,
        )
        live = LiveRun(id=run_id, request=request, progress=Progress(analysts=request.analysts))
        with self._lock:
            self._live[run_id] = live
        self._queue.put(live)
        return run_id

    def live(self, run_id: int) -> LiveRun | None:
        with self._lock:
            return self._live.get(run_id)

    def cancel(self, run_id: int) -> bool:
        live = self.live(run_id)
        if live is None or live.status not in ("queued", "running"):
            return False
        live.cancel.set()
        return True

    def _work(self) -> None:
        while True:
            live = self._queue.get()
            try:
                self.execute(live)
            except Exception:  # never let one run kill the worker
                logger.exception("dashboard run %s crashed", live.id)
            finally:
                with self._lock:
                    self._live.pop(live.id, None)

    def execute(self, live: LiveRun) -> None:
        request = live.request
        if live.cancel.is_set():
            self._finish(live, "cancelled", {})
            return
        live.status = "running"
        live.started = time.time()
        self.store.update(live.id, status="running", started_at=now_iso())
        live.progress.add_message("System", f"Analysing {request.ticker} for {request.trade_date}")

        final_state: dict = {}
        try:
            graph = self.graph_factory(request.analysts, config=build_config(request), callbacks=[live.stats])
            init = graph.create_run_state(request.ticker, request.trade_date, request.asset_type)
            args = graph.propagator.get_graph_args(callbacks=[live.stats])
            for messages, chunk in graph.stream_run(init, **args):
                if live.cancel.is_set():
                    raise Cancelled
                for message in messages:
                    kind, content = classify_message_type(message)
                    if content:
                        live.progress.add_message(kind, content, getattr(message, "id", None))
                    for call in getattr(message, "tool_calls", None) or []:
                        name = call["name"] if isinstance(call, dict) else call.name
                        live.progress.add_message("Tool", name, None)
                if chunk is not None:
                    live.progress.apply(chunk)
            final_state = live.progress.state
            graph.record_decision(request.ticker, request.trade_date, final_state)
            report = graph.save_reports(final_state, request.ticker)
            rating = run_rating(final_state)
            live.progress.add_message("System", f"Finished: {rating}")
            self._finish(live, "done", final_state, rating=rating, report_path=str(report.parent))
        except Cancelled:
            live.progress.add_message("System", "Stopped at your request")
            self._finish(live, "cancelled", live.progress.state)
        except Exception as exc:
            logger.exception("run %s failed", live.id)
            live.progress.add_message("System", f"Failed: {exc}")
            self._finish(live, "failed", live.progress.state, error=f"{type(exc).__name__}: {exc}")

    def _finish(self, live: LiveRun, status: str, state: dict, **fields) -> None:
        live.status = status
        live.error = fields.get("error")
        stats = live.stats.get_stats()
        self.store.save_snapshot(live.id, {
            "sections": sections(state, live.request.analysts),
            "messages": live.progress.messages,
            "elapsed": live.elapsed(),
        })
        self.store.update(
            live.id, status=status, finished_at=now_iso(),
            llm_calls=stats["llm_calls"], tool_calls=stats["tool_calls"],
            tokens_in=stats["tokens_in"], tokens_out=stats["tokens_out"], **fields,
        )
