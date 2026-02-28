---
description: Oracle Cloud resource limits, memory budget, multi-TF deployment policy
globs:
alwaysApply: false
---
# Skill: Oracle Cloud Resource Limits

> Loaded by DevOps before deploying new runners or adding streams.
> Last reviewed: 2026-02-27

## Hardware Spec

| Resource | Value |
|----------|-------|
| RAM | 1GB (Free Tier) |
| Swap | Configured (mandatory) |
| Boot Volume | ~46.6GB |
| OS | Ubuntu 22.04, x86_64 |

## Current Memory Budget (estimated, 2026-02-27)

| Process | Description | RAM |
|---------|-------------|-----|
| `meta_blend_live` | WebSocketRunner, 8 symbols x 1h | ~200-350MB |
| `tg_bot` | Telegram Bot (MultiStrategyBot) | ~50-80MB |
| OS + system | Ubuntu 22.04 baseline | ~200MB |
| Cron (transient) | Data downloads every 2-6h | ~100MB (short-lived) |
| **Total** | | **~450-630MB / 1024MB** |

~400MB available headroom after OI Liq Bounce shelved (2026-02-27).

## Multi-TF Deployment Policy

1. **Backtest / research (Multi-TF)**: Run on **local Mac only**, never on Oracle Cloud.
2. **Live deployment (Multi-TF features)**: **No additional WebSocket streams**.
   - Use `_resample_ohlcv()` from 1h data. Zero extra WS connections.
3. **Derivatives data (LSR/CVD)**: Use **cron + file-based refresh** (parquet, refreshes every 30min).
4. **If strategy truly needs real-time 5m data**: Escalate to upgrade discussion (2-4GB, ~$10-20/month).

## Disk Budget

| Data | Estimated Size |
|------|---------------|
| 1h klines x 8 symbols x 5+ years | ~50MB |
| 5m klines x 8 (research only) | ~400MB |
| Derivatives (LSR/CVD/Liq) | ~50-100MB |
| On-chain data | ~10-20MB |
| **Total** | **~460-520MB** |

46.6GB boot volume is more than sufficient.

## Pre-Deploy Resource Check (Guardrail)

Before deploying any new runner or adding WS streams:

1. SSH in: `free -h`
2. Estimate incremental RAM
3. Verify: **at least 150MB free after swap** (below → swap thrashing → latency degradation)
4. If insufficient: remove a runner, merge via `meta_blend`, or upgrade instance
