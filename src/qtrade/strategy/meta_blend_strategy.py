"""
Meta-Blend Strategy — 多策略信號混合器

動機：
    允許在同一個 Runner 中同時運行多個子策略，
    將它們的持倉信號加權混合後輸出單一信號。
    這樣可以：
    1. 避免多 Runner 搶倉位（ONE_WAY mode 限制）
    2. 自動享受策略分散化效益
    3. 用一套風控管理所有策略
    4. 支援 per-symbol 不同的子策略配置

架構：
    meta_blend 是一個已註冊的策略（@register_strategy），
    它內部呼叫其他已註冊的策略，加權混合信號。
    現有的 Runner 看到的只是一個普通策略 —— 零改動。

    ┌─────────────────────────────┐
    │      meta_blend             │
    │  ┌───────┐  ┌───────────┐  │
    │  │ sub_A  │  │   sub_B    │  │
    │  │(0.30)  │  │  (0.70)   │  │
    │  └───┬───┘  └─────┬─────┘  │
    │      └──────┬─────┘        │
    │             ▼              │
    │    weighted sum → clip     │
    │    → final position        │
    └─────────────────────────────┘

用法 (YAML config):
    strategy:
      name: "meta_blend"
      params:
        sub_strategies:
          - name: "tsmom_ema"
            weight: 0.30
            params: {lookback: 168, ...}
          - name: "tsmom_carry_v2"
            weight: 0.70
            params: {tier: "default", ...}

    也支援 symbol_overrides 讓每個幣種有不同的子策略組合：
      symbol_overrides:
        BTCUSDT:
          sub_strategies:
            - name: "breakout_vol_atr"
              weight: 0.30
              params: {...}
            - name: "tsmom_carry_v2"
              weight: 0.70
              params: {tier: "btc_enhanced", ...}

Note:
    meta_blend 使用 auto_delay=False 註冊，因為每個子策略透過
    get_strategy() 呼叫時已自帶 delay 和 direction clip。
    如果使用 auto_delay=True，會導致 auto_delay=False 的子策略
    （如 breakout_vol_atr，內建 delay）被雙重 delay。
"""
from __future__ import annotations

import pandas as pd
import logging

from .base import StrategyContext
from . import register_strategy, get_strategy

logger = logging.getLogger(__name__)


@register_strategy("meta_blend", auto_delay=False)
def generate_meta_blend(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    多策略信號混合策略

    內部呼叫 N 個子策略，將信號加權平均。

    params:
        sub_strategies: list[dict]
            每個 dict 包含：
                name:   str  — 子策略名稱（已註冊）
                weight: float — 權重（所有 weight 會自動正規化到 sum=1）
                params: dict  — 傳給子策略的參數

    Returns:
        加權混合後的持倉信號 [-1, 1]
    """
    sub_strategies = params.get("sub_strategies", [])
    if not sub_strategies:
        logger.error("meta_blend: 沒有定義 sub_strategies!")
        return pd.Series(0.0, index=df.index)

    # ── 正規化權重 ──
    total_w = sum(s.get("weight", 1.0) for s in sub_strategies)
    if total_w == 0:
        logger.error("meta_blend: 權重總和為 0!")
        return pd.Series(0.0, index=df.index)

    # ── 執行每個子策略 ──
    blended = pd.Series(0.0, index=df.index)
    for sub in sub_strategies:
        sub_name = sub["name"]
        sub_weight = sub.get("weight", 1.0) / total_w
        sub_params = dict(sub.get("params", {}))

        # 傳遞 _data_dir（如果有的話）
        if "_data_dir" in params and "_data_dir" not in sub_params:
            sub_params["_data_dir"] = params["_data_dir"]

        try:
            # 使用 wrapped strategy（含 delay/clip）—— 每個子策略
            # 各自處理自己的 delay 和 direction clip，meta_blend
            # 本身 auto_delay=False 不再加。
            sub_func = get_strategy(sub_name)
            sub_signal = sub_func(df, ctx, sub_params)
            blended += sub_weight * sub_signal
            logger.debug(
                f"meta_blend: {sub_name} w={sub_weight:.2f} "
                f"avg_pos={sub_signal.abs().mean():.3f}"
            )
        except Exception as e:
            logger.error(f"meta_blend: 子策略 {sub_name} 失敗: {e}")
            # 失敗的子策略信號為 0，權重自動落在其他策略上

    # ── Clip to [-1, 1] ──
    blended = blended.clip(-1.0, 1.0)

    n_subs = len(sub_strategies)
    sub_names = [s["name"] for s in sub_strategies]
    logger.info(
        f"meta_blend: {ctx.symbol} 混合 {n_subs} 個子策略 {sub_names} "
        f"avg_|pos|={blended.abs().mean():.3f}"
    )

    return blended
