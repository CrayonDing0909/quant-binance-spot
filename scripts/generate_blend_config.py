#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Meta-Blend Config 自動生成工具
═══════════════════════════════════════════════════════════════

從 compare_strategies.py 的輸出（或手動指定）自動生成
meta_blend YAML 配置檔，省去手動編寫。

用法：
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate

  # 從比較報告 JSON 自動生成
  PYTHONPATH=src python scripts/generate_blend_config.py \\
    --from-report reports/strategy_comparison/20260225_120000/strategy_comparison.json \\
    --output config/research_auto_blend.yaml

  # 手動指定策略組合
  PYTHONPATH=src python scripts/generate_blend_config.py \\
    --strategies config/prod_live_R3C_E3.yaml config/research_oi_liq_bounce.yaml \\
    --weights 0.70 0.30 \\
    --output config/research_auto_blend.yaml

  # 從比較報告 JSON 使用最佳化權重
  PYTHONPATH=src python scripts/generate_blend_config.py \\
    --from-report reports/strategy_comparison/20260225_120000/strategy_comparison.json \\
    --use-optimal-weights \\
    --output config/research_auto_blend.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from qtrade.config import load_config


def _load_comparison_report(report_path: str) -> dict:
    """載入 compare_strategies.py 的輸出 JSON"""
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_strategy_params(config_path: str) -> dict:
    """
    從 config YAML 讀取完整的 strategy params。
    處理 ensemble 路由和 symbol_overrides。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    strategy = raw.get("strategy", {})
    ensemble = raw.get("ensemble", {})

    return {
        "config_path": config_path,
        "strategy_name": strategy.get("name", "unknown"),
        "params": strategy.get("params", {}),
        "symbol_overrides": strategy.get("symbol_overrides", {}),
        "market": raw.get("market", {}),
        "futures": raw.get("futures", {}),
        "backtest": raw.get("backtest", {}),
        "portfolio": raw.get("portfolio", {}),
        "ensemble": ensemble,
    }


def generate_blend_from_report(
    report: dict,
    use_optimal_weights: bool = False,
) -> dict:
    """
    從比較報告自動生成 meta_blend 配置。

    Args:
        report: compare_strategies.py 的輸出 dict
        use_optimal_weights: 是否使用最佳化權重（否則用等權重）

    Returns:
        完整的 YAML 配置 dict
    """
    strategies = report.get("strategies", [])
    if len(strategies) < 2:
        raise ValueError("比較報告至少需要 2 個策略")

    # 策略權重
    if use_optimal_weights and "optimal_weights" in report:
        opt = report["optimal_weights"]
        if "optimal_weights" in opt:
            weights = opt["optimal_weights"]
        else:
            weights = {s["label"]: 1.0 / len(strategies) for s in strategies}
    else:
        weights = {s["label"]: 1.0 / len(strategies) for s in strategies}

    # 收集所有策略的參數
    strat_params = {}
    all_symbols = set()
    base_config = None

    for strat_info in strategies:
        cp = strat_info.get("config_path", "")
        if not cp or not Path(cp).exists():
            continue
        sp = _read_strategy_params(cp)
        strat_params[strat_info["label"]] = sp
        all_symbols.update(strat_info.get("symbols", []))
        if base_config is None:
            base_config = sp

    if not strat_params:
        raise ValueError("無法讀取任何策略配置")

    # 確定幣種聯集
    all_symbols = sorted(all_symbols)

    # 建構 meta_blend 配置
    # 使用第一個策略的 market/futures/backtest 作為基礎
    config = {
        "market": {
            "symbols": all_symbols,
            "interval": base_config["market"].get("interval", "1h"),
            "start": base_config["market"].get("start"),
            "end": base_config["market"].get("end"),
            "market_type": base_config["market"].get("market_type", "futures"),
        },
    }

    # Futures 配置
    if base_config.get("futures"):
        config["futures"] = {
            "leverage": base_config["futures"].get("leverage", 1),
            "margin_type": base_config["futures"].get("margin_type", "ISOLATED"),
            "position_mode": base_config["futures"].get("position_mode", "ONE_WAY"),
            "direction": base_config["futures"].get("direction", "both"),
        }

    # 建構 sub_strategies（預設組合 — 所有策略）
    sub_strategies = []
    for label, sp in strat_params.items():
        w = weights.get(label, 1.0 / len(strat_params))
        sub = {
            "name": sp["strategy_name"],
            "weight": round(w, 4),
            "params": sp["params"],
        }
        sub_strategies.append(sub)

    config["strategy"] = {
        "name": "meta_blend",
        "params": {
            "sub_strategies": sub_strategies,
        },
    }

    # ── 建構 symbol_overrides ──
    # 如果某個策略有 symbol_overrides，需要合併
    # 邏輯：對每個 symbol，檢查各策略是否有特殊配置
    symbol_overrides = {}
    for sym in all_symbols:
        sym_subs = []
        for label, sp in strat_params.items():
            w = weights.get(label, 1.0 / len(strat_params))
            # 檢查此策略是否包含這個 symbol
            strat_symbols = []
            for s_info in strategies:
                if s_info["label"] == label:
                    strat_symbols = s_info.get("symbols", [])
                    break

            if sym not in strat_symbols:
                continue

            # 檢查是否有 symbol-specific override
            if sym in sp.get("symbol_overrides", {}):
                override = sp["symbol_overrides"][sym]
                # 如果 override 裡有 sub_strategies（meta_blend 內的 sub），直接用
                if "sub_strategies" in override:
                    for sub in override["sub_strategies"]:
                        sym_subs.append({
                            "name": sub["name"],
                            "weight": round(sub.get("weight", 1.0) * w, 4),
                            "params": sub.get("params", {}),
                        })
                else:
                    # 普通 override：用策略名 + override 的 params
                    sym_params = dict(sp["params"])
                    sym_params.update(override)
                    sym_subs.append({
                        "name": sp["strategy_name"],
                        "weight": round(w, 4),
                        "params": sym_params,
                    })
            else:
                # 使用預設參數
                sym_subs.append({
                    "name": sp["strategy_name"],
                    "weight": round(w, 4),
                    "params": sp["params"],
                })

        # 如果此 symbol 的 sub_strategies 和預設不同，加入 overrides
        if sym_subs and sym_subs != sub_strategies:
            symbol_overrides[sym] = {
                "sub_strategies": sym_subs,
            }

    if symbol_overrides:
        config["strategy"]["symbol_overrides"] = symbol_overrides

    # Backtest 配置
    config["backtest"] = base_config.get("backtest", {
        "initial_cash": 10000,
        "fee_bps": 5,
        "slippage_bps": 3,
        "trade_on": "next_open",
        "validate_data": True,
        "clean_data": True,
        "funding_rate": {
            "enabled": True,
            "default_rate_8h": 0.0001,
            "use_historical": True,
        },
        "slippage_model": {
            "enabled": True,
        },
    })

    # Risk 配置
    config["risk"] = {"max_drawdown_pct": 0.40}

    # Position sizing
    config["position_sizing"] = {
        "method": "fixed",
        "position_pct": 1.0,
    }

    # Portfolio allocation（等權重或從最佳化來）
    n_syms = len(all_symbols)
    config["portfolio"] = {
        "cash_reserve": 0,
        "allocation": {sym: round(1.0 / n_syms, 4) for sym in all_symbols},
    }

    # Output
    config["output"] = {"report_dir": "./reports"}

    return config


def generate_blend_from_configs(
    config_paths: list[str],
    weights: list[float] | None = None,
) -> dict:
    """
    從多個 config 路徑直接生成 meta_blend 配置。
    """
    n = len(config_paths)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # 建構假的 report 格式以重用生成邏輯
    strategies = []
    for cp, w in zip(config_paths, weights):
        sp = _read_strategy_params(cp)
        cfg = load_config(cp)
        strategies.append({
            "label": sp["strategy_name"],
            "config_path": cp,
            "strategy_name": sp["strategy_name"],
            "symbols": cfg.market.symbols,
            "stats": {},
        })

    report = {
        "strategies": strategies,
        "optimal_weights": {
            "optimal_weights": {
                s["label"]: w for s, w in zip(strategies, weights)
            },
        },
    }

    return generate_blend_from_report(report, use_optimal_weights=True)


def write_blend_config(config: dict, output_path: str):
    """寫入 meta_blend YAML 配置"""
    header = f"""\
# ═══════════════════════════════════════════════════════════════
# AUTO-GENERATED — Meta-Blend 配置
# ═══════════════════════════════════════════════════════════════
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Tool: scripts/generate_blend_config.py
#
# 子策略:
"""
    subs = config.get("strategy", {}).get("params", {}).get("sub_strategies", [])
    for sub in subs:
        header += f"#   - {sub['name']} (weight={sub.get('weight', 0):.2f})\n"

    header += """\
#
# ⚠️  此配置為自動生成，請先回測驗證後再使用
# ═══════════════════════════════════════════════════════════════

"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(
            config, f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"✅ meta_blend 配置已生成: {output_path}")
    print(f"   子策略: {', '.join(s['name'] for s in subs)}")
    print(f"   幣種: {', '.join(config.get('market', {}).get('symbols', []))}")
    print()
    print(f"   下一步: 回測驗證")
    print(f"   PYTHONPATH=src python scripts/run_portfolio_backtest.py -c {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="從比較報告或手動指定生成 meta_blend YAML 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 兩種輸入模式
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--from-report", type=str,
        help="從 compare_strategies.py 的輸出 JSON 生成",
    )
    input_group.add_argument(
        "--strategies", nargs="+", type=str,
        help="手動指定策略 config 路徑列表",
    )

    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="策略權重（與 --strategies 搭配使用）",
    )
    parser.add_argument(
        "--use-optimal-weights", action="store_true",
        help="使用比較報告中的最佳化權重（與 --from-report 搭配使用）",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="輸出 YAML 路徑",
    )

    args = parser.parse_args()

    if args.from_report:
        report = _load_comparison_report(args.from_report)
        config = generate_blend_from_report(report, use_optimal_weights=args.use_optimal_weights)
    else:
        if args.weights and len(args.weights) != len(args.strategies):
            print(f"❌ --weights 數量與 --strategies 數量不符")
            sys.exit(1)
        config = generate_blend_from_configs(args.strategies, args.weights)

    write_blend_config(config, args.output)


if __name__ == "__main__":
    main()
