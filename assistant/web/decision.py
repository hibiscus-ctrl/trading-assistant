"""The run's decision in a few fields, read back from the structured agents' markdown.

The Portfolio Manager and the Trader fill typed schemas (PortfolioDecision,
TraderProposal) and the engine renders them to markdown with fixed ``**Label**:``
headers that its own parsers rely on (see ``tradingagents/agents/schemas.py``).
This reads those headers back, so the card shows exactly what the agents decided,
with no extra model call. A field the agent left out, or text that is not in the
structured shape, comes back as None and the card says so.
"""

from __future__ import annotations

import re

from tradingagents.agents.rating import run_rating

_FIELD = r"\*\*{label}\*\*:[ \t]*(.*?)(?=\n\s*\n\*\*[A-Z][A-Za-z ]+\*\*:|\n\s*FINAL TRANSACTION PROPOSAL|\Z)"
_MISSING = {"", "not provided", "none", "n/a", "null"}


def _field(text: str, label: str) -> str | None:
    match = re.search(_FIELD.format(label=re.escape(label)), text or "", re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value.lower() in _MISSING else value


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"[^\d\-]*(-?[\d,]*\.?\d+)\s*", value)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_decision(state: dict) -> dict | None:
    """The card's fields, or None when the run reached no final decision."""
    final = state.get("final_trade_decision") or ""
    if not final.strip():
        return None
    trader = state.get("trader_investment_plan") or ""
    entry, stop = _number(_field(trader, "Entry Price")), _number(_field(trader, "Stop Loss"))
    target = _number(_field(final, "Price Target"))
    return {
        "rating": run_rating(state),
        "summary": _field(final, "Executive Summary"),
        "thesis": _field(final, "Investment Thesis"),
        "price_target": target,
        "horizon": _field(final, "Time Horizon"),
        "trader_action": _field(trader, "Action"),
        "trader_reasoning": _field(trader, "Reasoning"),
        "entry": entry,
        "stop": stop,
        "sizing": _field(trader, "Position Sizing"),
        "upside_pct": round((target / entry - 1) * 100, 1) if target and entry else None,
        "risk_pct": round((stop / entry - 1) * 100, 1) if stop and entry else None,
    }


def one_line(decision: dict | None, limit: int = 160) -> str | None:
    """The first sentence or so of the summary, for the runs list."""
    if not decision or not decision.get("summary"):
        return None
    text = " ".join(decision["summary"].split())
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(first) > limit:
        first = first[: limit - 1].rsplit(" ", 1)[0] + "…"
    return first
