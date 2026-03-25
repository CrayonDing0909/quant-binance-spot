from qtrade.config import StrategyConfig, load_config


def test_strategy_config_get_params_returns_deepcopy():
    cfg = StrategyConfig(
        name="meta_blend",
        params={
            "sub_strategies": [
                {"name": "base", "params": {"window": 10}},
            ],
            "nested": {"threshold": 1},
        },
        symbol_overrides={
            "BTCUSDT": {
                "nested": {"threshold": 2},
            }
        },
    )

    params = cfg.get_params("BTCUSDT")
    params["sub_strategies"][0]["params"]["window"] = 99
    params["nested"]["threshold"] = 123

    assert cfg.params["sub_strategies"][0]["params"]["window"] == 10
    assert cfg.symbol_overrides["BTCUSDT"]["nested"]["threshold"] == 2


def test_load_config_uses_dataclass_defaults_for_optional_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
market:
  symbols: ["BTCUSDT"]
  interval: "1h"
  start: "2022-01-01"
  end: null
  market_type: "futures"

futures:
  leverage: 3

strategy:
  name: "test_strategy"
  params: {}

backtest:
  initial_cash: 10000
  fee_bps: 4
  slippage_bps: 3
  trade_on: "next_open"

output:
  report_dir: "./reports"
""".strip()
    )

    cfg = load_config(str(config_path))

    assert cfg.portfolio.cash_reserve == 0.2
    assert cfg.risk.max_drawdown_pct == 0.20
    assert cfg.futures is not None
    assert cfg.futures.direction == "both"
    assert cfg.live.limit_order_timeout_s == 10
    assert cfg.live.rebalance_band.threshold_pct == 0.03
    assert cfg.live.symbol_governance.thresholds.consistency_min == 99.0
    assert cfg.backtest.funding_rate.use_historical is True
    assert cfg.backtest.slippage_model.adv_lookback == 20
