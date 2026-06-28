from datetime import datetime

from scripts.run_dca import completed_key, period_key, spent_today, validate_limits, DCAConfig, DCAOrder


def _cfg(**overrides):
    data = {
        "dry_run": True,
        "schedule": "daily",
        "timezone": "Asia/Taipei",
        "state_path": "reports/dca/state.json",
        "quote_asset": "USDT",
        "min_quote_balance_after": 0.0,
        "max_total_quote_per_run": 50.0,
        "max_total_quote_per_day": 75.0,
        "redeem_flexible_before_order": False,
        "redeem_extra_quote": 0.0,
        "notify": False,
        "orders": [DCAOrder("BTCUSDT", 25.0), DCAOrder("ETHUSDT", 25.0)],
    }
    data.update(overrides)
    return DCAConfig(**data)


def test_period_key_daily_weekly_monthly():
    now = datetime(2026, 6, 28, 12, 0)
    assert period_key(now, "daily") == "2026-06-28"
    assert period_key(now, "weekly") == "2026-W26"
    assert period_key(now, "monthly") == "2026-06"


def test_completed_key_normalizes_symbol():
    assert completed_key("2026-06-28", "btcusdt") == "2026-06-28:BTCUSDT"


def test_spent_today_counts_only_live_runs_for_date():
    state = {
        "runs": [
            {"date": "2026-06-28", "quote_qty": 25, "live": True},
            {"date": "2026-06-28", "quote_qty": 25, "live": False},
            {"date": "2026-06-27", "quote_qty": 25, "live": True},
        ]
    }
    assert spent_today(state, "2026-06-28") == 25.0


def test_validate_limits_allows_config_within_caps():
    validate_limits(_cfg(), {"runs": [{"date": "2026-06-28", "quote_qty": 25, "live": True}]}, "2026-06-28")


def test_validate_limits_rejects_run_cap():
    cfg = _cfg(max_total_quote_per_run=40.0)
    try:
        validate_limits(cfg, {"runs": []}, "2026-06-28")
    except ValueError as exc:
        assert "max_total_quote_per_run" in str(exc)
    else:
        raise AssertionError("expected ValueError")
