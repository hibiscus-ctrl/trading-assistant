"""A run's progress, built from the chunks ``TradingAgentsGraph.stream_run`` yields.

Pure functions over the merged state, so the live view, the saved snapshot and the
tests all read progress the same way. Mirrors what ``cli/run.py`` shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

ANALYST_ORDER = ("market", "social", "news", "fundamentals")
ANALYST_LABELS = {
    "market": "Market",
    "social": "Sentiment",
    "news": "News",
    "fundamentals": "Fundamentals",
}
ANALYST_REPORTS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

MAX_MESSAGES = 300
MAX_MESSAGE_CHARS = 4000


@dataclass
class Progress:
    analysts: list[str]
    state: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    _seen_ids: set = field(default_factory=set)

    def apply(self, chunk: dict) -> None:
        """Merge one streamed chunk: analysts' report deltas and full top-level states alike."""
        self.state.update(chunk)

    def add_message(self, kind: str, content: str, seen_id: str | None = None) -> None:
        if seen_id is not None:
            if seen_id in self._seen_ids:
                return
            self._seen_ids.add(seen_id)
        text = content.strip()
        if not text:
            return
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + " …"
        self.messages.append({"time": datetime.now().strftime("%H:%M:%S"), "kind": kind, "text": text})
        if len(self.messages) > MAX_MESSAGES:
            del self.messages[: len(self.messages) - MAX_MESSAGES]


def sections(state: dict, analysts: list[str]) -> list[dict]:
    """The report sections in reading order, each with its text so far (may be empty)."""
    out = [
        {"key": ANALYST_REPORTS[a], "title": f"{ANALYST_LABELS[a]} analyst", "text": _text(state.get(ANALYST_REPORTS[a]))}
        for a in ANALYST_ORDER if a in analysts
    ]
    out.append({"key": "investment_plan", "title": "Research debate", "text": _research(state)})
    out.append({"key": "trader_investment_plan", "title": "Trader", "text": _text(state.get("trader_investment_plan"))})
    out.append({"key": "final_trade_decision", "title": "Risk debate & final decision", "text": _risk(state)})
    return out


def stages(state: dict, analysts: list[str], active: bool) -> list[dict]:
    """Pipeline steps with a status each: pending, running or done."""
    steps = []
    for a in ANALYST_ORDER:
        if a not in analysts:
            continue
        done = bool(state.get(ANALYST_REPORTS[a]))
        steps.append({"name": ANALYST_LABELS[a], "status": "done" if done else ("running" if active else "pending")})
    analysts_done = all(s["status"] == "done" for s in steps)

    debate = state.get("investment_debate_state") or {}
    research_done = bool(state.get("investment_plan"))
    research_started = bool((debate.get("bull_history") or "").strip() or (debate.get("bear_history") or "").strip())
    steps.append(_step("Bull ⇄ Bear", research_done, research_started or (analysts_done and active)))

    trader_done = bool(state.get("trader_investment_plan"))
    steps.append(_step("Trader", trader_done, research_done and active))

    risk = state.get("risk_debate_state") or {}
    risk_started = any((risk.get(k) or "").strip() for k in ("aggressive_history", "conservative_history", "neutral_history"))
    decision_done = bool(state.get("final_trade_decision"))
    steps.append(_step("Risk debate", decision_done, risk_started or (trader_done and active)))
    steps.append(_step("Portfolio manager", decision_done, False))
    if not active:
        for step in steps:
            if step["status"] == "running":
                step["status"] = "pending"
    return steps


def current_section(state: dict, analysts: list[str]) -> dict | None:
    """The latest section with text, for the live view's 'current report' panel."""
    filled = [s for s in sections(state, analysts) if s["text"]]
    return filled[-1] if filled else None


def _step(name: str, done: bool, started: bool) -> dict:
    return {"name": name, "status": "done" if done else ("running" if started else "pending")}


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value).strip()
    return str(value).strip()


def _research(state: dict) -> str:
    debate = state.get("investment_debate_state") or {}
    parts = []
    if (bull := _text(debate.get("bull_history"))):
        parts.append(f"### Bull researcher\n{bull}")
    if (bear := _text(debate.get("bear_history"))):
        parts.append(f"### Bear researcher\n{bear}")
    if (judge := _text(state.get("investment_plan"))):
        parts.append(f"### Research manager decision\n{judge}")
    return "\n\n".join(parts)


def _risk(state: dict) -> str:
    risk = state.get("risk_debate_state") or {}
    parts = []
    for key, label in (("aggressive_history", "Aggressive analyst"),
                       ("conservative_history", "Conservative analyst"),
                       ("neutral_history", "Neutral analyst")):
        if (text := _text(risk.get(key))):
            parts.append(f"### {label}\n{text}")
    if (judge := _text(state.get("final_trade_decision"))):
        parts.append(f"### Portfolio manager decision\n{judge}")
    return "\n\n".join(parts)
