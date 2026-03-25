"""
Quant Trading Dashboard — FastAPI + Tailwind CSS
Lightweight, mobile-responsive, deployable on Oracle Cloud.

Run locally:
    PYTHONPATH=src python scripts/dashboard_web.py

Deploy on Oracle Cloud:
    PYTHONPATH=src nohup python scripts/dashboard_web.py --host 0.0.0.0 --port 8501 &
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

app = FastAPI(title="交易指揮中心")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# ── Data loaders ─────────────────────────────────────────────────────────────


def load_tasks() -> list[dict]:
    tasks = []
    for p in sorted((ROOT / "tasks" / "active").glob("*.yaml")):
        try:
            with open(p) as f:
                t = yaml.safe_load(f)
            if t:
                t["_file"] = p.name
                tasks.append(t)
        except Exception:
            pass
    return tasks


def load_equity(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT date, equity, pnl_day, cumulative_pnl, drawdown_pct "
            "FROM daily_equity ORDER BY date DESC LIMIT 60",
            con,
        )
        if df.empty:
            # Fallback: derive from trades (simplified config era)
            trades = pd.read_sql(
                """SELECT timestamp, pnl, fee FROM trades
                   WHERE pnl IS NOT NULL AND symbol NOT IN ('ADAUSDT','BNBUSDT')
                   ORDER BY id""",
                con,
            )
            con.close()
            if trades.empty:
                return []
            trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
            cutoff = pd.Timestamp("2026-03-04 07:00:00", tz="UTC")
            trades = trades[trades["timestamp"] >= cutoff]
            if trades.empty:
                return []
            trades["date"] = trades["timestamp"].dt.normalize()
            daily = trades.groupby("date").agg(
                pnl_day=("pnl", "sum"), fee_day=("fee", "sum")
            ).reset_index()
            daily["pnl_day"] = daily["pnl_day"] - daily["fee_day"]
            daily["cumulative_pnl"] = daily["pnl_day"].cumsum()
            daily["equity"] = 10000 + daily["cumulative_pnl"]
            peak = daily["equity"].cummax()
            daily["drawdown_pct"] = ((daily["equity"] - peak) / peak * 100).round(2)
            daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
            return daily[["date", "equity", "pnl_day", "cumulative_pnl", "drawdown_pct"]].to_dict("records")
        else:
            con.close()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df.sort_values("date").to_dict("records")
    except Exception:
        return []


def load_recent_trades(db_path: Path, limit: int = 20) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql(
            f"SELECT timestamp, symbol, side, qty, price, pnl, reason "
            f"FROM trades ORDER BY id DESC LIMIT {limit}",
            con,
        )
        con.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m-%d %H:%M")
        df["pnl"] = df["pnl"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        return df.to_dict("records")
    except Exception:
        return []


def load_prod_config() -> dict:
    for name in ["prod_candidate_simplified.yaml", "prod_candidate_meta_blend.yaml"]:
        p = ROOT / "config" / name
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    return {}


def find_live_dbs() -> list[Path]:
    return sorted((ROOT / "reports").rglob("trading.db"))


def load_symbol_rankings(db_path: Path) -> list[dict]:
    """Per-symbol performance for the strategy ranking table."""
    if not db_path or not db_path.exists():
        return []
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql(
            """SELECT symbol, side, pnl, fee FROM trades
               WHERE symbol NOT IN ('ADAUSDT','BNBUSDT')
               ORDER BY id""",
            con,
        )
        con.close()
        if df.empty:
            return []

        results = []
        for sym, grp in df.groupby("symbol"):
            closed = grp[grp["pnl"].notna()]
            total_trades = len(closed)
            if total_trades == 0:
                continue
            net_pnl = closed["pnl"].sum() - closed["fee"].sum()
            wins = (closed["pnl"] > 0).sum()
            wr = wins / total_trades * 100 if total_trades > 0 else 0
            # Simple max drawdown from cumulative pnl
            cum = (closed["pnl"] - closed["fee"]).cumsum()
            peak = cum.cummax()
            dd = (cum - peak).min()
            results.append({
                "symbol": sym.replace("USDT", ""),
                "trades": total_trades,
                "wr": round(wr, 1),
                "pnl": round(net_pnl, 2),
                "dd": round(dd, 2),
            })
        return sorted(results, key=lambda x: x["pnl"], reverse=True)
    except Exception:
        return []


def load_reports() -> list[dict]:
    """Scan reports/ for recent validation/research outputs."""
    reports = []
    for d in sorted((ROOT / "reports").iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        # Find markdown reports
        for md in sorted(d.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:2]:
            try:
                first_line = md.read_text().split("\n", 1)[0].strip("# ").strip()
            except Exception:
                first_line = md.stem
            reports.append({
                "category": d.name,
                "title": first_line[:60],
                "path": str(md.relative_to(ROOT)),
            })
        if len(reports) >= 8:
            break
    return reports


# ── API endpoints ────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    tasks = load_tasks()
    cfg = load_prod_config()
    dbs = find_live_dbs()
    db = dbs[0] if dbs else None

    # Primary DB: prefer meta_blend
    for d in dbs:
        if "meta_blend" in str(d):
            db = d
            break

    equity = load_equity(db) if db else []
    trades = load_recent_trades(db, limit=15) if db else []
    rankings = load_symbol_rankings(db)
    reports = load_reports()

    running = sum(1 for t in tasks if t.get("status") == "running")
    needs_review = sum(
        1 for t in tasks
        if t.get("review_required") or (t.get("approval") or {}).get("required")
    )
    research_count = sum(1 for _ in (ROOT / "config").glob("research_*.yaml"))
    blocked = sum(1 for t in tasks if t.get("status") == "blocked")

    # Process tasks for template
    stage_order = [
        "alpha_research", "quant_developer", "validation",
        "risk_review", "risk_manager", "deployment", "stop_or_handoff", "complete",
    ]
    for t in tasks:
        stage = t.get("current_stage", "")
        idx = stage_order.index(stage) if stage in stage_order else 0
        t["_pct"] = int((idx / max(len(stage_order) - 1, 1)) * 100)
        t["_short_name"] = t.get("task_id", "").split("_", 3)[-1].replace("_", " ")[:40]
        fp = t.get("final_packet") or {}
        rec = (fp.get("recommendation") or "")
        findings = t.get("research_findings") or {}
        outcome = findings.get("final_outcome") or ""
        combined = rec + outcome
        if "WEAK GO" in combined:
            t["_verdict"] = "WEAK GO"
        elif "FAIL" in combined:
            t["_verdict"] = "FAIL"
        elif "GO" in combined:
            t["_verdict"] = "GO"
        else:
            t["_verdict"] = ""

    # Decisions
    decisions = [
        t for t in tasks
        if t.get("review_required") or (t.get("approval") or {}).get("required")
    ]

    # Latest findings
    all_findings = []
    for t in tasks:
        f = t.get("research_findings") or {}
        for sig in (f.get("signals_tested") or [])[-3:]:
            all_findings.append({"task": t["_short_name"], "signal": sig})

    context = {
        "request": request,
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "running": running,
        "needs_review": needs_review,
        "research_count": research_count,
        "blocked": blocked,
        "tasks": tasks,
        "equity": equity,
        "trades": trades,
        "decisions": decisions,
        "findings": all_findings,
        "symbols": cfg.get("symbols", []),
        "interval": cfg.get("interval", "—"),
        "leverage": cfg.get("leverage", "—"),
        "research_cfgs": [p.name for p in sorted((ROOT / "config").glob("research_*.yaml"))],
        "rankings": rankings,
        "reports": reports,
        "completed_count": sum(1 for t in tasks if t.get("status") == "completed"),
    }
    return templates.TemplateResponse(request, "dashboard.html", context=context)


@app.get("/api/equity")
async def api_equity():
    dbs = find_live_dbs()
    db = None
    for d in dbs:
        if "meta_blend" in str(d):
            db = d
            break
    if not db and dbs:
        db = dbs[0]
    return JSONResponse(load_equity(db) if db else [])


@app.get("/api/trades")
async def api_trades():
    dbs = find_live_dbs()
    db = None
    for d in dbs:
        if "meta_blend" in str(d):
            db = d
            break
    if not db and dbs:
        db = dbs[0]
    return JSONResponse(load_recent_trades(db) if db else [])


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Quant Dashboard Web Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    print(f"Dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
