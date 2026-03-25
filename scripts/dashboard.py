"""
Quant Trading Team Dashboard
Human-in-the-loop oversight panel for agent team and live strategies.

Run:
    PYTHONPATH=src streamlit run scripts/dashboard.py
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="交易指揮中心",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1rem; padding-bottom: 0rem; }
  .metric-card {
    background: #1e1e2e; border-radius: 8px; padding: 12px 16px;
    border-left: 4px solid #7c3aed; margin-bottom: 8px;
  }
  .agent-card {
    background: #1a1a2e; border-radius: 6px; padding: 10px 12px; margin-bottom: 6px;
  }
  .status-running  { color: #22c55e; font-weight: bold; }
  .status-pending  { color: #f59e0b; font-weight: bold; }
  .status-blocked  { color: #ef4444; font-weight: bold; }
  .status-done     { color: #6b7280; }
  .decision-card {
    background: #2d1b1b; border: 1px solid #dc2626; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 8px;
  }
  .go-badge        { background: #166534; color: #86efac; border-radius: 4px; padding: 2px 8px; font-size: 12px; }
  .fail-badge      { background: #7f1d1d; color: #fca5a5; border-radius: 4px; padding: 2px 8px; font-size: 12px; }
  .weak-badge      { background: #713f12; color: #fde68a; border-radius: 4px; padding: 2px 8px; font-size: 12px; }
  .section-header  { font-size: 14px; font-weight: 600; color: #a78bfa; margin-bottom: 6px; }
  div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
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
            pass  # skip malformed yaml
    return tasks


@st.cache_data(ttl=60)
def load_equity_from_db(db_path: Path, days: int = 60) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(db_path)
        # Try daily_equity first
        df = pd.read_sql(
            f"SELECT date, equity, pnl_day, cumulative_pnl, drawdown_pct "
            f"FROM daily_equity ORDER BY date DESC LIMIT {days}",
            con,
        )
        if not df.empty:
            con.close()
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date")
        # Fallback: derive daily equity from trades PnL
        # Filter to simplified config era (after last ADA/BNB trade = htf_lsr era)
        trades = pd.read_sql(
            """SELECT timestamp, pnl, fee FROM trades
               WHERE pnl IS NOT NULL
                 AND symbol NOT IN ('ADAUSDT','BNBUSDT')
               ORDER BY id""",
            con,
        )
        con.close()
        if trades.empty:
            return pd.DataFrame()
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        # Only keep trades from simplified era (Mar 4 onwards)
        cutoff = pd.Timestamp("2026-03-04 07:00:00", tz="UTC")
        trades = trades[trades["timestamp"] >= cutoff]
        if trades.empty:
            return pd.DataFrame()
        trades["date"] = trades["timestamp"].dt.normalize()
        daily = trades.groupby("date").agg(pnl_day=("pnl", "sum"), fee_day=("fee", "sum")).reset_index()
        daily["pnl_day"] = daily["pnl_day"] - daily["fee_day"]
        initial_cash = 10000.0
        daily["cumulative_pnl"] = daily["pnl_day"].cumsum()
        daily["equity"] = initial_cash + daily["cumulative_pnl"]
        peak = daily["equity"].cummax()
        daily["drawdown_pct"] = ((daily["equity"] - peak) / peak * 100).round(2)
        daily["date"] = daily["date"].dt.tz_localize(None)
        return daily[["date", "equity", "pnl_day", "cumulative_pnl", "drawdown_pct"]].tail(days)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_recent_trades(db_path: Path, limit: int = 20) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql(
            f"SELECT timestamp, symbol, side, qty, price, pnl, reason "
            f"FROM trades ORDER BY id DESC LIMIT {limit}",
            con,
        )
        con.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_prod_config() -> dict:
    cfg_path = ROOT / "config" / "prod_candidate_simplified.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "config" / "prod_candidate_meta_blend.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def find_live_dbs() -> list[Path]:
    return sorted((ROOT / "reports").rglob("trading.db"))


def stage_emoji(stage: str) -> str:
    return {
        "alpha_research": "🔬",
        "quant_developer": "⚙️",
        "validation": "✅",
        "risk_review": "🛡️",
        "risk_manager": "🛡️",
        "stop_or_handoff": "🏁",
        "deployment": "🚀",
        "complete": "🏁",
        "blocked": "🚧",
    }.get(stage, "📋")


def status_color(status: str) -> str:
    return {
        "running": "status-running",
        "pending": "status-pending",
        "blocked": "status-blocked",
        "complete": "status-done",
        "paused": "status-pending",
    }.get(status, "")


# ── Header ────────────────────────────────────────────────────────────────────

tasks = load_tasks()
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

running = sum(1 for t in tasks if t.get("status") == "running")
blocked = sum(1 for t in tasks if t.get("status") == "blocked")
needs_review = sum(1 for t in tasks if t.get("review_required") or t.get("approval", {}).get("required"))
research_count = sum(1 for p in (ROOT / "config").glob("research_*.yaml"))

col_title, col_m1, col_m2, col_m3, col_m4, col_time = st.columns([3, 1, 1, 1, 1, 2])
with col_title:
    st.markdown("## 🎯 交易指揮中心")
with col_m1:
    st.metric("執行中", running)
with col_m2:
    st.metric("待審核", needs_review, delta="⚠️" if needs_review > 0 else None)
with col_m3:
    st.metric("研究中", research_count)
with col_m4:
    st.metric("封鎖", blocked, delta="🔴" if blocked > 0 else None)
with col_time:
    st.markdown(f"<div style='text-align:right;color:#6b7280;padding-top:14px'>{now_str}</div>", unsafe_allow_html=True)

st.divider()

# ── Main layout: 4 columns ────────────────────────────────────────────────────

col_agents, col_research, col_strategy, col_decisions = st.columns([1.1, 1.1, 1.4, 1.1])

# ── Column 1: Agent Tasks ─────────────────────────────────────────────────────

with col_agents:
    st.markdown('<div class="section-header">🤖 Agent 任務</div>', unsafe_allow_html=True)

    if not tasks:
        st.caption("目前沒有進行中的任務")
    else:
        for task in tasks:
            tid = task.get("task_id", task.get("_file", ""))
            status = task.get("status", "unknown")
            stage = task.get("current_stage", "")
            goal = task.get("goal", "")
            owner = task.get("owner_agent", "")
            updated = task.get("updated_at", "")
            review_req = task.get("review_required") or task.get("approval", {}).get("required", False)

            sc = status_color(status)
            short_goal = goal[:80] + "…" if len(goal) > 80 else goal

            badge = ""
            if review_req:
                badge = ' <span style="background:#7c3aed;color:white;border-radius:3px;padding:1px 6px;font-size:11px">待審</span>'

            st.markdown(
                f"""<div class="agent-card">
                <div style="font-size:13px;font-weight:600">{stage_emoji(stage)} {tid.split("_", 3)[-1].replace("_", " ")[:40]}{badge}</div>
                <div style="font-size:11px;color:#9ca3af;margin-top:3px">{short_goal}</div>
                <div style="font-size:11px;margin-top:5px">
                  <span class="{sc}">{status.upper()}</span>
                  &nbsp;·&nbsp;<span style="color:#6b7280">{owner}</span>
                  &nbsp;·&nbsp;<span style="color:#4b5563">{stage}</span>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header" style="margin-top:12px">📂 Research Configs</div>', unsafe_allow_html=True)
    research_cfgs = sorted((ROOT / "config").glob("research_*.yaml"))
    for p in research_cfgs:
        st.markdown(f"<div style='font-size:11px;color:#a78bfa;padding:2px 0'>📄 {p.name}</div>", unsafe_allow_html=True)

# ── Column 2: Research Pipeline ───────────────────────────────────────────────

with col_research:
    st.markdown('<div class="section-header">🔬 研究進度</div>', unsafe_allow_html=True)

    stage_order = ["alpha_research", "quant_developer", "validation", "risk_review", "risk_manager", "deployment", "stop_or_handoff", "complete"]

    for task in tasks:
        tid = task.get("task_id", "")
        stage = task.get("current_stage", "")
        status = task.get("status", "")
        final = task.get("final_packet", {})
        recommendation = (final.get("recommendation") or "") if final else ""
        findings = task.get("research_findings") or {}
        outcome = findings.get("final_outcome") or ""
        short_name = tid.split("_", 3)[-1].replace("_", " ")[:35]

        # Stage progress bar
        idx = stage_order.index(stage) if stage in stage_order else 0
        pct = int((idx / (len(stage_order) - 1)) * 100)

        verdict_badge = ""
        if "WEAK GO" in (recommendation + outcome):
            verdict_badge = '<span class="weak-badge">WEAK GO</span>'
        elif "GO" in (recommendation + outcome) and "FAIL" not in (recommendation + outcome):
            verdict_badge = '<span class="go-badge">GO</span>'
        elif "FAIL" in (recommendation + outcome):
            verdict_badge = '<span class="fail-badge">FAIL</span>'

        st.markdown(
            f"""<div class="agent-card">
            <div style="font-size:12px;font-weight:600">{short_name} {verdict_badge}</div>
            <div style="background:#374151;border-radius:4px;height:4px;margin:6px 0">
              <div style="background:#7c3aed;width:{pct}%;height:4px;border-radius:4px"></div>
            </div>
            <div style="font-size:11px;color:#9ca3af">{stage_emoji(stage)} {stage} · {pct}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # PR workflow reminder
    st.markdown('<div class="section-header" style="margin-top:14px">🔀 GitHub PRs</div>', unsafe_allow_html=True)
    st.caption("每個研究應建立獨立 PR branch")
    st.code("git checkout -b research/<name>-$(date +%Y%m%d)\ngh pr create --label research", language="bash")

# ── Column 3: Live Strategy Performance ──────────────────────────────────────

with col_strategy:
    st.markdown('<div class="section-header">📊 策略績效</div>', unsafe_allow_html=True)

    dbs = find_live_dbs()
    cfg = load_prod_config()
    prod_symbols = cfg.get("symbols", [])

    if not dbs:
        st.warning("沒有找到 trading.db，請確認 live runner 有在跑")
    else:
        db_choice = st.selectbox(
            "資料庫",
            dbs,
            format_func=lambda p: str(p.relative_to(ROOT / "reports")),
            label_visibility="collapsed",
        )

        equity_df = load_equity_from_db(db_choice, days=60)
        trades_df = load_recent_trades(db_choice, limit=50)

        if not equity_df.empty:
            latest = equity_df.iloc[-1]
            e0 = equity_df["equity"].iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Equity", f"${latest['equity']:,.0f}")
            ret_pct = (latest["equity"] / e0 - 1) * 100 if e0 > 0 else 0
            m2.metric("總收益", f"{ret_pct:+.1f}%")
            m3.metric("今日 PnL", f"${latest['pnl_day']:+.0f}")
            m4.metric("最大 DD", f"{equity_df['drawdown_pct'].min():.1f}%")

            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df["date"], y=equity_df["equity"],
                fill="tozeroy", line=dict(color="#7c3aed", width=2),
                fillcolor="rgba(124,58,237,0.15)", name="Equity",
            ))
            fig.update_layout(
                height=180, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="#374151", color="#6b7280"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Recent trades table
        st.markdown('<div class="section-header">最近交易</div>', unsafe_allow_html=True)
        if not trades_df.empty:
            display_df = trades_df[["timestamp", "symbol", "side", "price", "pnl"]].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%m-%d %H:%M")
            display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "")
            st.dataframe(
                display_df,
                use_container_width=True,
                height=160,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("時間", width=90),
                    "symbol": st.column_config.TextColumn("幣種", width=80),
                    "side": st.column_config.TextColumn("方向", width=55),
                    "price": st.column_config.NumberColumn("價格", format="%.2f", width=80),
                    "pnl": st.column_config.TextColumn("PnL", width=70),
                },
            )

    # Prod config summary
    if cfg:
        st.markdown('<div class="section-header" style="margin-top:8px">⚙️ 生產配置</div>', unsafe_allow_html=True)
        symbols = cfg.get("symbols", [])
        interval = cfg.get("interval", "?")
        leverage = cfg.get("leverage", "?")
        st.markdown(
            f"<div style='font-size:12px;color:#d1d5db'>"
            f"📅 {interval} &nbsp;·&nbsp; ⚡ {leverage}x &nbsp;·&nbsp; "
            f"{'  '.join(symbols)}</div>",
            unsafe_allow_html=True,
        )

# ── Column 4: Decisions + Reports ────────────────────────────────────────────

with col_decisions:
    st.markdown('<div class="section-header">⚠️ 待你決策</div>', unsafe_allow_html=True)

    decision_items = []
    for task in tasks:
        approval = task.get("approval", {}) or {}
        review_req = task.get("review_required", False)
        if approval.get("required") or review_req:
            decision_items.append(task)

    if not decision_items:
        st.markdown(
            "<div style='color:#22c55e;font-size:13px;padding:8px 0'>✅ 全部已讀，無待決策項目</div>",
            unsafe_allow_html=True,
        )
    else:
        for task in decision_items:
            tid = task.get("task_id", "")
            short_name = tid.split("_", 3)[-1].replace("_", " ")[:30]
            reason = (task.get("approval", {}) or {}).get("reason", task.get("review_reason", "需要人工審核"))
            fp = task.get("final_packet", {}) or {}
            key_decision = fp.get("key_decision", "")
            rec_agent = fp.get("recommended_agent", "")

            st.markdown(
                f"""<div class="decision-card">
                <div style="font-size:13px;font-weight:600;color:#fca5a5">🚨 {short_name}</div>
                <div style="font-size:11px;color:#d1d5db;margin-top:4px">{reason}</div>
                {"<div style='font-size:11px;color:#fde68a;margin-top:4px'>" + key_decision[:100] + "…</div>" if key_decision else ""}
                {"<div style='font-size:11px;color:#9ca3af;margin-top:4px'>→ " + rec_agent + "</div>" if rec_agent else ""}
                </div>""",
                unsafe_allow_html=True,
            )

    # Findings from tasks
    st.markdown('<div class="section-header" style="margin-top:14px">🔍 最新發現</div>', unsafe_allow_html=True)
    for task in tasks:
        findings = task.get("research_findings", {}) or {}
        signals_tested = findings.get("signals_tested", [])
        if signals_tested:
            short_name = task.get("task_id", "").split("_", 3)[-1].replace("_", " ")[:25]
            st.markdown(f"<div style='font-size:11px;color:#a78bfa;margin:4px 0;font-weight:600'>{short_name}</div>", unsafe_allow_html=True)
            for sig in signals_tested[-3:]:
                verdict = "✅" if "GO" in sig and "FAIL" not in sig else "❌" if "FAIL" in sig else "⚠️"
                short_sig = sig[:70] + "…" if len(sig) > 70 else sig
                st.markdown(f"<div style='font-size:11px;color:#9ca3af;padding:1px 0'>{verdict} {short_sig}</div>", unsafe_allow_html=True)

    # Reports
    st.markdown('<div class="section-header" style="margin-top:14px">📁 歷史報告</div>', unsafe_allow_html=True)
    report_dirs = sorted([p for p in (ROOT / "reports").iterdir() if p.is_dir()], reverse=True)
    for p in report_dirs[:6]:
        st.markdown(f"<div style='font-size:11px;color:#6b7280;padding:2px 0'>📂 {p.name}</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.caption("👤 **人類責任**：GO/NO-GO 決策、資本配置、風險上限")
with fc2:
    st.caption("🤖 **Agent 責任**：研究執行、回測、驗證、程式碼審查")
with fc3:
    if st.button("🔄 重新整理", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
