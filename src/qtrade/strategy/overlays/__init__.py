"""
Overlay 模組 — 策略後處理層（風控 / 出場 / 降倉 / 加速 / 擇時縮放）

設計理念：
    Overlay 不改變策略的進場邏輯，只在持倉後根據額外市場資料
    （如 OI、波動率、LSR、微結構等）進行「減倉 / 平倉 / 暫停新加倉 / 加速 / 擇時縮放」。

可用 Overlay：
    1. oi_vol_exit_overlay  — OI + Vol 出場（R2 系列）
    2. microstructure_accel_overlay — 5m/15m 微結構加速層（R3 Track A）
    3. lsr_confirmatory_overlay — 散戶 LSR 擇時縮放（Confirmatory Scale）
    4. derivatives_micro_overlay — 衍生品微觀結構增強

支援複合模式（'+' 連鎖）：
    mode: "vol_pause+lsr_confirmatory" → 先 vol_pause 再 LSR 縮放

用法：
    from qtrade.strategy.overlays.oi_vol_exit_overlay import (
        compute_oi_signals,
        compute_vol_state,
        apply_oi_vol_exit_overlay,
        apply_overlay_by_mode,  # 中央 dispatcher（支援複合模式）
    )

    from qtrade.strategy.overlays.lsr_confirmatory_overlay import (
        apply_lsr_confirmatory_overlay,
        compute_lsr_pctrank,
    )

    from qtrade.strategy.overlays.microstructure_accel_overlay import (
        compute_micro_features,
        compute_accel_score,
        apply_accel_overlay,
        apply_full_micro_accel_overlay,
    )
"""
