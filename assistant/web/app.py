"""The dashboard's routes: start an analysis, watch it live, browse past runs.

Served on 127.0.0.1 only; open it through VS Code's Ports tab. Pages use a
relative <base href>, so they also work behind code-server's /proxy/<port>/ path.
State-changing requests must come from htmx (``HX-Request`` header): a browser
cannot add that header cross-site without a CORS preflight we never allow, so
another website cannot start paid runs through your browser.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from assistant.web.progress import ANALYST_LABELS, ANALYST_ORDER, current_section, sections, stages
from assistant.web.runner import RunManager, RunRequest
from assistant.web.store import ACTIVE, RunStore
from cli.models import AssetType
from cli.prefs import load_last_run
from cli.prompts import _llm_provider_table, detect_asset_type, parse_analysis_date, parse_ticker
from tradingagents.default_config import _TRADINGAGENTS_HOME
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

HERE = Path(__file__).parent
DEFAULT_DATA_DIR = Path(_TRADINGAGENTS_HOME) / "web"

# Model output is untrusted: CommonMark with raw HTML and images off, as report_html does.
_markdown = MarkdownIt("commonmark", {"html": False}).disable("image")


def render_markdown(text: str) -> str:
    return _markdown.render(text or "")


def base_href(path: str) -> str:
    """Relative path from this page back to the app root."""
    depth = len([part for part in path.split("/") if part])
    return "../" * (depth - 1) if depth > 1 else "./"


def available_providers() -> list[dict]:
    """Providers you can run now: an API key is set, or none is needed (e.g. Ollama)."""
    out = []
    for label, key, url in _llm_provider_table():
        if key not in MODEL_OPTIONS:
            continue
        env = get_api_key_env(key)
        if key == "ollama":
            if not os.environ.get("OLLAMA_BASE_URL"):
                continue  # offered only once a local model server is configured
        elif env is None:
            continue  # signed-in or cloud-credential providers stay in the terminal CLI
        if env and not os.environ.get(env):
            continue
        out.append({"key": key, "label": label, "url": url})
    return out


def model_choices(provider: str) -> dict[str, list[tuple[str, str]]]:
    options = MODEL_OPTIONS.get(provider, {})
    return {mode: [(label, value) for label, value in options.get(mode, []) if value != "custom"]
            for mode in ("quick", "deep")}


def create_app(data_dir: Path = DEFAULT_DATA_DIR, manager: RunManager | None = None) -> FastAPI:
    store = manager.store if manager else RunStore(data_dir)
    manager = manager or RunManager(store)
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["markdown"] = render_markdown
    app = FastAPI(title="Trading assistant", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.state.manager = manager

    def page(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"base": base_href(request.url.path), **context})

    def fragment(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def require_htmx(request: Request) -> None:
        if request.headers.get("HX-Request") != "true":
            raise HTTPException(status_code=403, detail="Use the dashboard page to do this.")

    def runs_context() -> dict:
        runs = store.recent()
        return {"runs": runs, "any_active": any(r["status"] in ACTIVE for r in runs)}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        prefs = load_last_run()
        providers = available_providers()
        keys = [p["key"] for p in providers]
        provider = prefs.get("llm_provider") if prefs.get("llm_provider") in keys else (keys[0] if keys else None)
        same = provider == prefs.get("llm_provider")
        defaults = {
            "ticker": "",
            "date": date.today().isoformat(),
            "analysts": prefs.get("analysts") or list(ANALYST_ORDER),
            "depth": prefs.get("research_depth") or 1,
            "provider": provider,
            "quick_model": prefs.get("quick_think_llm") if same else "",
            "deep_model": prefs.get("deep_think_llm") if same else "",
        }
        if provider and not defaults["quick_model"]:
            choices = model_choices(provider)
            defaults["quick_model"] = choices["quick"][0][1] if choices["quick"] else ""
            defaults["deep_model"] = choices["deep"][0][1] if choices["deep"] else ""
        return page(request, "index.html", providers=providers, defaults=defaults,
                    choices=model_choices(provider) if provider else {"quick": [], "deep": []},
                    analyst_labels=ANALYST_LABELS, analyst_order=ANALYST_ORDER, **runs_context())

    @app.get("/models", response_class=HTMLResponse)
    def models(request: Request, provider: str):
        return fragment(request, "_model_fields.html", choices=model_choices(provider),
                        defaults={"quick_model": "", "deep_model": ""}, fresh=True)

    @app.get("/runs/table", response_class=HTMLResponse)
    def runs_table(request: Request):
        return fragment(request, "_runs_table.html", **runs_context())

    @app.post("/runs")
    def start_run(request: Request, ticker: str = Form(...), trade_date: str = Form(...),
                  provider: str = Form(...), quick_model: str = Form(...), deep_model: str = Form(...),
                  depth: int = Form(1), analysts: Annotated[list[str] | None, Form()] = None):
        require_htmx(request)
        errors = []
        try:
            ticker = parse_ticker(ticker)
        except ValueError as exc:
            errors.append(str(exc))
        try:
            trade_date = parse_analysis_date(trade_date)
        except ValueError as exc:
            errors.append(str(exc))
        providers = {p["key"]: p for p in available_providers()}
        if provider not in providers:
            errors.append(f"No API key set for provider {provider!r}")
        picked = [a for a in ANALYST_ORDER if a in (analysts or [])]
        if not picked:
            errors.append("Pick at least one analyst")
        if not quick_model.strip() or not deep_model.strip():
            errors.append("Choose both models")
        if not 1 <= depth <= 5:
            errors.append("Debate rounds must be between 1 and 5")
        if errors:
            return fragment(request, "_errors.html", errors=errors)
        asset_type = detect_asset_type(ticker)
        if asset_type == AssetType.CRYPTO:
            picked = [a for a in picked if a != "fundamentals"] or ["market"]
        run_id = manager.submit(RunRequest(
            ticker=ticker, trade_date=trade_date, asset_type=asset_type.value, analysts=picked,
            provider=provider, backend_url=providers[provider]["url"],
            quick_model=quick_model.strip(), deep_model=deep_model.strip(), depth=depth,
        ))
        return Response(status_code=204, headers={"HX-Redirect": f"runs/{run_id}"})

    def run_view(run_id: int) -> dict:
        record = store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="No such run")
        live = manager.live(run_id)
        if live is not None and live.status in ACTIVE:
            stats = live.stats.get_stats()
            state = live.progress.state
            return {
                "run": record, "live": True, "status": live.status,
                "stages": stages(state, record["analysts"], live.status == "running"),
                "current": current_section(state, record["analysts"]),
                "messages": list(reversed(live.progress.messages[-60:])),
                "stats": stats, "elapsed": int(live.elapsed()),
            }
        snapshot = store.load_snapshot(run_id) or {}
        done_sections = snapshot.get("sections") or sections({}, record["analysts"])
        filled = {s["key"] for s in done_sections if s["text"]}
        return {
            "run": record, "live": False, "status": record["status"],
            "stages": [{"name": s["title"], "status": "done" if s["key"] in filled else "pending"}
                       for s in done_sections],
            "sections": done_sections,
            "messages": list(reversed(snapshot.get("messages") or [])),
            "stats": {k: record[k] for k in ("llm_calls", "tool_calls", "tokens_in", "tokens_out")},
            "elapsed": int(snapshot.get("elapsed") or 0),
            "has_html": bool(record["report_path"]) and (Path(record["report_path"]) / "complete_report.html").is_file(),
        }

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: int):
        return page(request, "run.html", **run_view(run_id))

    @app.get("/runs/{run_id}/live", response_class=HTMLResponse)
    def run_live(request: Request, run_id: int):
        view = run_view(run_id)
        if not view["live"]:
            # The run just ended: reload the page to show the finished report.
            return Response(status_code=200, headers={"HX-Refresh": "true"})
        return fragment(request, "_live.html", **view)

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(request: Request, run_id: int):
        require_htmx(request)
        manager.cancel(run_id)
        return Response(status_code=204)

    @app.get("/runs/{run_id}/report")
    def run_report(run_id: int):
        record = store.get(run_id)
        if not record or not record["report_path"]:
            raise HTTPException(status_code=404, detail="No report for this run")
        page_path = Path(record["report_path"]) / "complete_report.html"
        if not page_path.is_file():
            raise HTTPException(status_code=404, detail="The HTML report was not saved")
        return FileResponse(page_path, media_type="text/html")

    @app.get("/tickers/{ticker}", response_class=HTMLResponse)
    def ticker_page(request: Request, ticker: str):
        runs = store.for_ticker(ticker.upper())
        if not runs:
            raise HTTPException(status_code=404, detail="No runs for this ticker")
        return page(request, "ticker.html", ticker=ticker.upper(), runs=runs)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading assistant dashboard")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TA_WEB_PORT", "8765")))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    import uvicorn

    # Loopback only: there is no login, so the dashboard must not be reachable from the network.
    uvicorn.run(create_app(args.data_dir), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
