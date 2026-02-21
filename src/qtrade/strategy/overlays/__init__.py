"""
Overlay 模組 — 策略後處理層（風控 / 出場 / 降倉）

設計理念：
    Overlay 不改變策略的進場邏輯，只在持倉後根據額外市場資料
    （如 OI、波動率等）進行「減倉 / 平倉 / 暫停新加倉」。

用法：
    from qtrade.strategy.overlays.oi_vol_exit_overlay import (
        compute_oi_signals,
        compute_vol_state,
        apply_oi_vol_exit_overlay,
    )
"""
