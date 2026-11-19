"""The local dashboard (assistant/web): progress reading, the run lifecycle, and its guards."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from assistant.web.app import base_href, create_app, render_markdown
from assistant.web.progress import Progress, current_section, sections, stages
from assistant.web.runner import RunManager
from assistant.web.store import RunStore

pytestmark = pytest.mark.unit

HTMX = {"HX-Request": "true"}
FORM = {
    "ticker": "nvda", "trade_date": "2026-11-18", "provider": "openai",
    "quick_model": "gpt-6-luna", "deep_model": "gpt-6-sol", "depth": "1",
    "analysts": ["market", "news"],
}


class FakeGraph:
    """Streams a tiny run the way TradingAgentsGraph.stream_run does, without any model."""

    def __init__(self, analysts, config=None, callbacks=None, *, report_dir: Path, on_step=None):
        self.analysts, self.config, self.report_dir, self.on_step = analysts, config, report_dir, on_step
        self.recorded = None

        class _Propagator:
            def get_graph_args(self, callbacks=None):
                return {}

        self.propagator = _Propagator()

    def create_run_state(self, ticker, trade_date, asset_type="stock", portfolio=None):
        return {"company_of_interest": ticker, "trade_date": trade_date}

    def stream_run(self, graph_input, **args):
        steps = [
            ([AIMessage(content="Looking at the chart", id="m1")], None),
            ([AIMessage(content="", id="m2", tool_calls=[{"name": "get_stock_data", "args": {}, "id": "t1"}])], None),
            ([], {"market_report": "Strong trend"}),
            ([], {"news_report": "Quiet week"}),
            ([], {"investment_debate_state": {"bull_history": "Bull says up", "bear_history": "Bear says down"},
                  "investment_plan": "Lean long"}),
            ([], {"trader_investment_plan": "Buy a starter position"}),
            ([], {"risk_debate_state": {"aggressive_history": "More", "conservative_history": "Less",
                                        "neutral_history": "Some"},
                  "final_trade_decision": "Rating: Buy", "final_rating": "Buy"}),
        ]
        for step in steps:
            if self.on_step:
                self.on_step()
            yield step

    def record_decision(self, ticker, trade_date, final_state):
        self.recorded = (ticker, trade_date, dict(final_state))

    def save_reports(self, final_state, ticker, save_path=None, html=True):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "complete_report.html").write_text("<p>report</p>", encoding="utf-8")
        return self.report_dir / "complete_report.md"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TA_WEB_ALLOWED_HOSTS", "testserver")  # TestClient's Host header
    monkeypatch.setattr("assistant.web.app.load_last_run", lambda: {})
    store = RunStore(tmp_path / "web")
    graphs = []

    def factory(analysts, config=None, callbacks=None):
        graph = FakeGraph(analysts, config, callbacks, report_dir=tmp_path / "report")
        graphs.append(graph)
        return graph

    manager = RunManager(store, graph_factory=factory, start_worker=False)
    client = TestClient(create_app(manager=manager))
    return client, manager, store, graphs


def test_progress_follows_the_pipeline():
    p = Progress(analysts=["market", "news"])
    names = lambda steps: {s["name"]: s["status"] for s in steps}  # noqa: E731
    assert names(stages(p.state, p.analysts, True)) == {
        "Market": "running", "News": "running", "Bull ⇄ Bear": "pending", "Trader": "pending",
        "Risk debate": "pending", "Portfolio manager": "pending",
    }
    p.apply({"market_report": "a"})
    p.apply({"news_report": "b"})
    assert names(stages(p.state, p.analysts, True))["Bull ⇄ Bear"] == "running"
    p.apply({"investment_debate_state": {"bull_history": "up"}, "investment_plan": "long"})
    assert names(stages(p.state, p.analysts, True))["Trader"] == "running"
    assert current_section(p.state, p.analysts)["key"] == "investment_plan"
    research = next(s for s in sections(p.state, p.analysts) if s["key"] == "investment_plan")
    assert "Bull researcher" in research["text"] and "Research manager decision" in research["text"]


def test_messages_dedupe_and_cap():
    p = Progress(analysts=["market"])
    p.add_message("Agent", "hello", "id1")
    p.add_message("Agent", "hello again", "id1")
    p.add_message("Agent", "   ")
    assert [m["text"] for m in p.messages] == ["hello"]


def test_run_lifecycle(env):
    client, manager, store, graphs = env
    response = client.post("/runs", data=FORM, headers=HTMX)
    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "runs/1"

    live = manager._queue.get_nowait()
    assert live.request.ticker == "NVDA" and live.request.analysts == ["market", "news"]
    assert client.get("/runs/1").status_code == 200  # queued page renders
    manager.execute(live)

    record = store.get(1)
    assert record["status"] == "done" and record["rating"] == "Buy"
    assert graphs[0].config["max_debate_rounds"] == 1 and graphs[0].config["checkpoint_enabled"] is False
    assert graphs[0].recorded[2]["final_trade_decision"] == "Rating: Buy"

    page = client.get("/runs/1").text
    assert "Buy a starter position" in page and "get_stock_data" in page
    assert client.get("/runs/1/report").text == "<p>report</p>"
    assert client.get("/runs/1/live").headers.get("HX-Refresh") == "true"
    assert "NVDA" in client.get("/tickers/nvda").text
    assert "runs/1" in client.get("/").text


def test_cancel_stops_between_steps(env):
    client, manager, store, graphs = env
    client.post("/runs", data=FORM, headers=HTMX)
    live = manager._queue.get_nowait()
    calls = {"n": 0}

    def factory(analysts, config=None, callbacks=None):
        def step():
            calls["n"] += 1
            if calls["n"] == 3:
                manager.cancel(live.id)
        return FakeGraph(analysts, config, callbacks, report_dir=Path("/nonexistent"), on_step=step)

    manager.graph_factory = factory
    live.status = "running"
    manager.execute(live)
    assert store.get(live.id)["status"] == "cancelled"


def test_state_changes_need_htmx(env):
    client, manager, store, _ = env
    assert client.post("/runs", data=FORM).status_code == 403
    assert client.post("/runs/1/cancel").status_code == 403
    assert store.recent() == []


def test_foreign_host_is_refused(env):
    """DNS rebinding: a page on evil.example re-resolved to 127.0.0.1 is same-origin, so it
    can send HX-Request; only the Host check stops it."""
    client, manager, store, _ = env
    response = client.post("/runs", data=FORM, headers={**HTMX, "Host": "evil.example:8765"})
    assert response.status_code == 400
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert store.recent() == []
    for host in ("localhost:8765", "127.0.0.1:8765", "[::1]:8765"):
        assert client.get("/", headers={"Host": host}).status_code == 200


def test_bad_input_is_reported_not_queued(env):
    client, manager, store, _ = env
    bad = {**FORM, "ticker": "../etc", "trade_date": "2999-01-01", "analysts": [], "provider": "nope"}
    text = client.post("/runs", data=bad, headers=HTMX).text
    assert "not a ticker symbol" in text and "in the future" in text
    assert "Pick at least one analyst" in text and "No API key set" in text
    assert store.recent() == []


def test_restart_marks_unfinished_runs_failed(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create(ticker="NVDA", trade_date="2026-11-18", provider="openai",
                          quick_model="q", deep_model="d", analysts=["market"], depth=1)
    store.update(run_id, status="running")
    assert RunStore(tmp_path).get(run_id)["status"] == "failed"


def test_base_href_and_markdown_safety():
    assert base_href("/") == "./"
    assert base_href("/runs/5") == "../"
    assert base_href("/runs/5/live") == "../../"
    html = render_markdown("<script>alert(1)</script> ![x](http://evil/x.png) [a](javascript:alert(1))")
    assert "<script" not in html and "<img" not in html and 'href="javascript' not in html


def test_dashboard_runs_retry_model_calls(env, monkeypatch):
    from assistant.web.runner import DASHBOARD_LLM_RETRIES, RunRequest, build_config
    from tradingagents.default_config import DEFAULT_CONFIG

    request = RunRequest("NVDA", "2026-11-18", "stock", ["market"], "openai", None, "q", "d", 1)
    monkeypatch.setitem(DEFAULT_CONFIG, "llm_max_retries", None)
    assert build_config(request)["llm_max_retries"] == DASHBOARD_LLM_RETRIES
    monkeypatch.setitem(DEFAULT_CONFIG, "llm_max_retries", 1)
    assert build_config(request)["llm_max_retries"] == 1  # an explicit setting wins


def test_rerun_copies_a_failed_run(env):
    client, manager, store, _ = env
    client.post("/runs", data=FORM, headers=HTMX)
    first = manager._queue.get_nowait()
    store.update(first.id, status="failed", error="OpenAIAPIError: 503")
    first.status = "failed"
    assert "Run again" in client.get(f"/runs/{first.id}").text
    assert client.post(f"/runs/{first.id}/rerun").status_code == 403
    response = client.post(f"/runs/{first.id}/rerun", headers=HTMX)
    assert response.headers["HX-Redirect"] == "runs/2"
    again = manager._queue.get_nowait().request
    assert (again.ticker, again.analysts, again.deep_model) == ("NVDA", ["market", "news"], "gpt-6-sol")
