# Exit Philosophy Research (2026-02-23)

## 假設
TSMOM-EMA 是動量策略，其收益結構可能以「少數大趨勢」驅動（右尾驅動）。
如果屬實，設置固定 TP 會截斷右尾、降低績效。
反之，若策略具有均值回歸特性，固定 TP 能提高勝率和穩定性。

## 三種哲學

### 1. Trend-hold (TH)
- 不設 TP，僅保留寬 SL（災難型保護 4-6x ATR）
- 出場靠策略信號反轉
- 假設：動量策略的利潤來自少數大趨勢，不應人為截斷

### 2. Hybrid-lock (HL)
- 不設硬 TP 上限
- 加入 trailing stop（2.0-3.5x ATR）鎖住已有收益
- 假設：保護浮盈可降低 MDD，同時不限制上行空間

### 3. Mean-revert-take (MR)
- 固定 TP（2.0-4.0x ATR）+ SL（2.0-3.0x ATR）
- 明確出場節奏
- 假設：crypto 短期有均值回歸特性，明確 TP 可提高勝率

## 基準
B0 = 現行裸 TSMOM-EMA（無 exit rules overlay），策略信號自行出場

## 參數矩陣
見 `scripts/research_exit_philosophy.py` 中的 `build_param_grid()`

## 執行
```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/research_exit_philosophy.py
```

## 結果
見 `reports/research/exit_philosophy/<timestamp>/`
