---
description: Binance Futures API gotchas (Hedge Mode, positionSide, order types, backtest vs live differences)
globs:
alwaysApply: false
---
# Skill: Binance API Gotchas

> Loaded by Quant Developer when working with Binance Futures API or broker code.

## Hedge Mode (對沖模式)

Account may be **Hedge Mode** (simultaneous long + short) or **One-Way Mode** (single direction).

**Critical**: Hedge Mode orders MUST specify `positionSide`:

```python
# Open Long
params = {"symbol": symbol, "side": "BUY", "positionSide": "LONG", "type": "MARKET", ...}

# Open Short
params = {"side": "SELL", "positionSide": "SHORT", ...}

# Close Long
params = {"side": "SELL", "positionSide": "LONG", ...}

# Close Short
params = {"side": "BUY", "positionSide": "SHORT", ...}
```

Error `-4061: Order's position side does not match user's setting` = missing `positionSide`.

## Order Response Type

Add `newOrderRespType=RESULT` to get fill info immediately:
```python
params = {..., "newOrderRespType": "RESULT"}  # Returns executedQty, avgPrice
```
Default `ACK` mode only returns orderId without fill details.

## SL/TP Order Types

| Type | Order Type | Use Case |
|------|-----------|----------|
| Stop Loss | `STOP_MARKET` | Capital protection, guaranteed fill, may slip |
| Take Profit | `TAKE_PROFIT_MARKET` | Lock profit, may slip |
| Take Profit (limit) | `TAKE_PROFIT` | More precise, but may not fill |

## Backtest vs Live Differences

| Item | Backtest | Live |
|------|---------|------|
| **SL/TP calc** | `entry ± N × ATR` | `entry ± N × ATR` ✅ consistent |
| **Trigger check** | Kline High/Low | Exchange Mark Price (real-time) |
| **Execution timing** | Within kline | Real-time trigger |
| **Execution price** | Assumes exact fill | May slip |
| **Trailing stop** | ✅ Supported | ❌ Not implemented |

## Backtest Return Expectations

⚠️ **Backtest returns need discount!**

| Reason | Impact |
|--------|--------|
| Slippage underestimated | Backtest 0.03%, actual 0.1-0.5% |
| Market impact | Large orders move price |
| Liquidity | May not fill at desired price |
| Capacity limit | $10K-$10M feasible, beyond that problematic |

**Rule of thumb**: `Actual return ≈ Backtest return ÷ 3~5`

Backtest value is in **comparing strategies**, not predicting absolute returns.
