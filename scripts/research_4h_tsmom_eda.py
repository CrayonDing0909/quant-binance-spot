#!/usr/bin/env python3
"""
4h TSMOM 時間框架優化 — Alpha EDA
==================================

Portfolio Context:
- 目標缺口: Alpha 覆蓋地圖中的「TF 優化」維度
- 整合目標: 若 IC 更強 → 替換主 TF；若互補 → 新 leg
- 5 因子評分: 分散化=3, 數據=5, Alpha=3, 複雜度=4, 文獻=4 → 總分 3.6

Analyses:
1. 1h vs 4h TSMOM IC / Rank IC (8 symbols)
2. 4h signal stability (yearly IC)
3. Correlation: pure 4h vs 1h+4h mixed (HTF Filter)
4. Cost impact: turnover comparison
5. 4h lookback sensitivity sweep

Usage:
    PYTHONPATH=src python scripts/research_4h_tsmom_eda.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
           'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT']
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'binance', 'futures', '1h')


# ══════════════════════════════════════════════════════════════
#  Core Functions
# ══════════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Causal resample (closed='left', label='left')"""
    return pd.DataFrame({
        'open': df['open'].resample(freq, closed='left', label='left').first(),
        'high': df['high'].resample(freq, closed='left', label='left').max(),
        'low': df['low'].resample(freq, closed='left', label='left').min(),
        'close': df['close'].resample(freq, closed='left', label='left').last(),
        'volume': df['volume'].resample(freq, closed='left', label='left').sum(),
    }).dropna(subset=['open', 'close'])


def compute_tsmom_signal(
    close: pd.Series,
    lookback: int = 168,
    vol_target: float = 0.15,
    ema_fast: int = 20,
    ema_slow: int = 50,
    annualize_factor: float = np.sqrt(8760),
) -> pd.Series:
    """Pure TSMOM + EMA alignment signal (mirrors production)"""
    returns = close.pct_change()
    cum_ret = returns.rolling(lookback).sum()
    vol = returns.rolling(lookback).std() * annualize_factor
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    raw_signal = np.sign(cum_ret)
    scale = (vol_target / vol).clip(0.1, 2.0)
    tsmom = (raw_signal * scale).clip(-1.0, 1.0).fillna(0.0)

    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    ema_trend = pd.Series(0.0, index=close.index)
    ema_trend[ema_f > ema_s] = 1.0
    ema_trend[ema_f < ema_s] = -1.0

    agree = np.sign(tsmom) == np.sign(ema_trend)
    pos = tsmom.copy()
    pos[~agree] *= 0.3

    return pos.clip(-1.0, 1.0)


def compute_ic(signal: pd.Series, fwd_ret: pd.Series) -> float:
    valid = signal.notna() & fwd_ret.notna() & (signal != 0)
    if valid.sum() < 100:
        return np.nan
    return signal[valid].corr(fwd_ret[valid])


def compute_rank_ic(signal: pd.Series, fwd_ret: pd.Series) -> float:
    valid = signal.notna() & fwd_ret.notna() & (signal != 0)
    if valid.sum() < 100:
        return np.nan
    return signal[valid].rank().corr(fwd_ret[valid].rank())


def yearly_ic(signal: pd.Series, fwd_ret: pd.Series) -> dict:
    result = {}
    for yr in sorted(signal.index.year.unique()):
        mask = signal.index.year == yr
        result[yr] = compute_ic(signal[mask], fwd_ret.reindex(signal[mask].index))
    return result


def get_1h_params(sym: str) -> dict:
    if sym == 'BTCUSDT':
        return {'lookback': 720, 'ema_fast': 30, 'ema_slow': 100}
    return {'lookback': 168, 'ema_fast': 20, 'ema_slow': 50}


def get_4h_default_params(sym: str) -> dict:
    """
    4h 等效參數：
    - lookback: 1h bars / 4  (168→42, 720→180)
    - EMA: 同比例縮減
    """
    if sym == 'BTCUSDT':
        return {'lookback': 180, 'ema_fast': 8, 'ema_slow': 25}
    return {'lookback': 42, 'ema_fast': 5, 'ema_slow': 12}


def make_1h_signal(close_1h: pd.Series, sym: str) -> pd.Series:
    p = get_1h_params(sym)
    return compute_tsmom_signal(close_1h, annualize_factor=np.sqrt(8760), **p)


def make_4h_signal(close_4h: pd.Series, sym: str) -> pd.Series:
    p = get_4h_default_params(sym)
    return compute_tsmom_signal(close_4h, annualize_factor=np.sqrt(2190), **p)


def compute_turnover_metrics(signal: pd.Series, bars_per_year: int) -> dict:
    pos_change = signal.diff().abs()
    turnover_per_bar = pos_change.mean()
    turnover_annual = turnover_per_bar * bars_per_year

    direction = np.sign(signal)
    dir_changes = (direction.diff().abs() > 0).sum()
    total_bars = len(signal)
    dir_changes_annual = dir_changes / total_bars * bars_per_year
    avg_hold_bars = total_bars / dir_changes if dir_changes > 0 else total_bars

    rt_cost = 0.0012  # 0.12%
    annual_cost_drag = turnover_annual * rt_cost

    return {
        'turnover_annual': turnover_annual,
        'dir_changes_annual': dir_changes_annual,
        'avg_hold_bars': avg_hold_bars,
        'annual_cost_pct': annual_cost_drag * 100,
    }


# ══════════════════════════════════════════════════════════════
#  HTF Filter Simulation
# ══════════════════════════════════════════════════════════════

def simulate_htf_filter(raw_pos: pd.Series, df_1h: pd.DataFrame) -> pd.Series:
    """Apply HTF filter (C_4h+daily_hard) to raw 1h signal"""
    from qtrade.strategy.multi_tf_resonance_strategy import _htf_trend, _daily_regime

    df_4h_r = resample_ohlcv(df_1h, '4h')
    df_1d_r = resample_ohlcv(df_1h, '1D')

    if len(df_4h_r) < 50 or len(df_1d_r) < 14:
        return raw_pos

    htf_4h_trend = _htf_trend(df_4h_r, 20, 50)
    htf_4h_aligned = htf_4h_trend.reindex(df_1h.index, method='ffill').fillna(0)

    regime = _daily_regime(df_1d_r, 14, 25.0, 20)
    daily_dir = regime['regime_direction'].reindex(df_1h.index, method='ffill').fillna(0)
    daily_trending = (regime['regime_trend'].reindex(df_1h.index, method='ffill').fillna(0) > 0.5)

    pos_dir = np.sign(raw_pos)
    htf_4h_dir = np.sign(htf_4h_aligned)

    htf_agree = (pos_dir == htf_4h_dir) & (pos_dir != 0) & (htf_4h_dir != 0)
    daily_agree = (pos_dir == daily_dir) & (pos_dir != 0) & (daily_dir != 0)

    confirmation = pd.Series(0.0, index=df_1h.index)
    confirmation[htf_agree & ~daily_agree] = 0.7
    confirmation[htf_agree & daily_agree & ~daily_trending] = 0.85
    confirmation[htf_agree & daily_agree & daily_trending] = 1.0

    return raw_pos * confirmation


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    # ── Load Data ──
    print('\n' + '='*80)
    print('  4h TSMOM 時間框架優化 — Alpha EDA')
    print('='*80)
    print('\n## 2. Data Description\n')

    data_1h = {}
    data_4h = {}
    for sym in SYMBOLS:
        fpath = os.path.join(DATA_DIR, f'{sym}.parquet')
        df = pd.read_parquet(fpath)
        data_1h[sym] = df
        data_4h[sym] = resample_ohlcv(df, '4h')
        print(f'  {sym}: 1h={len(df):>6,} bars ({df.index[0].date()} ~ {df.index[-1].date()}), '
              f'4h={len(data_4h[sym]):>5,} bars')

    # ══════════════════════════════════════════════════════════
    # 3. IC / Rank IC Comparison
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  3. IC / Rank IC Comparison — 1h vs 4h TSMOM')
    print('='*80)

    # 4h lookback variants
    VARIANTS_DEFAULT = {
        '4h_42bar_7d': {'lookback': 42, 'ema_fast': 5, 'ema_slow': 12},
        '4h_84bar_14d': {'lookback': 84, 'ema_fast': 10, 'ema_slow': 25},
        '4h_180bar_30d': {'lookback': 180, 'ema_fast': 20, 'ema_slow': 50},
    }
    VARIANTS_BTC = {
        '4h_180bar_30d': {'lookback': 180, 'ema_fast': 8, 'ema_slow': 25},
        '4h_360bar_60d': {'lookback': 360, 'ema_fast': 15, 'ema_slow': 50},
        '4h_504bar_84d': {'lookback': 504, 'ema_fast': 20, 'ema_slow': 65},
    }

    results_ic = []

    for sym in SYMBOLS:
        df_1h_sym = data_1h[sym]
        df_4h_sym = data_4h[sym]

        # 1h signal
        sig_1h = make_1h_signal(df_1h_sym['close'], sym)
        fwd_1h_24h = df_1h_sym['close'].pct_change(24).shift(-24)
        fwd_1h_4h = df_1h_sym['close'].pct_change(4).shift(-4)

        results_ic.append({
            'symbol': sym, 'config': '1h_prod',
            'IC_4h_fwd': compute_ic(sig_1h, fwd_1h_4h),
            'IC_24h_fwd': compute_ic(sig_1h, fwd_1h_24h),
            'RankIC_24h': compute_rank_ic(sig_1h, fwd_1h_24h),
        })

        # 4h variants
        variants = VARIANTS_BTC if sym == 'BTCUSDT' else VARIANTS_DEFAULT
        for cfg_name, cfg in variants.items():
            sig_4h = compute_tsmom_signal(
                df_4h_sym['close'], annualize_factor=np.sqrt(2190), **cfg,
            )
            fwd_4h_1bar = df_4h_sym['close'].pct_change(1).shift(-1)
            fwd_4h_6bar = df_4h_sym['close'].pct_change(6).shift(-6)

            results_ic.append({
                'symbol': sym, 'config': cfg_name,
                'IC_4h_fwd': compute_ic(sig_4h, fwd_4h_1bar),
                'IC_24h_fwd': compute_ic(sig_4h, fwd_4h_6bar),
                'RankIC_24h': compute_rank_ic(sig_4h, fwd_4h_6bar),
            })

    df_ic = pd.DataFrame(results_ic)

    # Pivot for IC_24h
    pivot_24h = df_ic.pivot(index='symbol', columns='config', values='IC_24h_fwd')
    pivot_ric = df_ic.pivot(index='symbol', columns='config', values='RankIC_24h')

    print('\n--- IC (Pearson) on 24h Forward Return ---')
    print(pivot_24h.round(4).to_string())
    print(f'\nMean IC:')
    print(pivot_24h.mean().round(4).to_string())

    print('\n--- Rank IC (Spearman) on 24h Forward Return ---')
    print(pivot_ric.round(4).to_string())
    print(f'\nMean Rank IC:')
    print(pivot_ric.mean().round(4).to_string())

    # Find best 4h config per symbol
    print('\n--- Best 4h Config per Symbol ---')
    for sym in SYMBOLS:
        rows_4h = df_ic[(df_ic['symbol'] == sym) & (df_ic['config'] != '1h_prod')]
        row_1h = df_ic[(df_ic['symbol'] == sym) & (df_ic['config'] == '1h_prod')]
        if len(rows_4h) > 0 and len(row_1h) > 0:
            best = rows_4h.loc[rows_4h['IC_24h_fwd'].idxmax()]
            ic_1h = row_1h['IC_24h_fwd'].values[0]
            delta = best['IC_24h_fwd'] - ic_1h
            winner = '4h ✅' if delta > 0 else '1h ✅'
            print(f'  {sym}: 1h IC={ic_1h:.4f}, best 4h={best["config"]} IC={best["IC_24h_fwd"]:.4f}, '
                  f'Δ={delta:+.4f} → {winner}')

    # ══════════════════════════════════════════════════════════
    # 4. Yearly IC Stability
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  4. Yearly IC Stability (24h Forward Return)')
    print('='*80)

    yearly_rows = []
    for sym in SYMBOLS:
        sig_1h = make_1h_signal(data_1h[sym]['close'], sym)
        sig_4h = make_4h_signal(data_4h[sym]['close'], sym)
        fwd_1h = data_1h[sym]['close'].pct_change(24).shift(-24)
        fwd_4h = data_4h[sym]['close'].pct_change(6).shift(-6)

        yic_1h = yearly_ic(sig_1h, fwd_1h)
        yic_4h = yearly_ic(sig_4h, fwd_4h)

        for yr in sorted(set(list(yic_1h.keys()) + list(yic_4h.keys()))):
            yearly_rows.append({
                'symbol': sym, 'year': yr,
                'IC_1h': yic_1h.get(yr, np.nan),
                'IC_4h': yic_4h.get(yr, np.nan),
            })

    df_yearly = pd.DataFrame(yearly_rows)

    for sym in SYMBOLS:
        sub = df_yearly[df_yearly['symbol'] == sym][['year', 'IC_1h', 'IC_4h']].set_index('year')
        flips_1h = (np.sign(sub['IC_1h'].dropna()).diff().abs() > 0).sum()
        flips_4h = (np.sign(sub['IC_4h'].dropna()).diff().abs() > 0).sum()
        print(f'\n  {sym} — sign flips: 1h={flips_1h}, 4h={flips_4h}')
        print(sub.round(4).to_string())

    print('\n--- Average IC Across 8 Symbols by Year ---')
    avg_yearly = df_yearly.groupby('year')[['IC_1h', 'IC_4h']].mean()
    print(avg_yearly.round(4).to_string())

    print('\n--- ICIR (mean/std across years, per symbol) ---')
    icir_rows = []
    for sym in SYMBOLS:
        sub = df_yearly[df_yearly['symbol'] == sym]
        icir_1h = sub['IC_1h'].mean() / sub['IC_1h'].std() if sub['IC_1h'].std() > 0 else 0
        icir_4h = sub['IC_4h'].mean() / sub['IC_4h'].std() if sub['IC_4h'].std() > 0 else 0
        icir_rows.append({'symbol': sym, 'ICIR_1h': icir_1h, 'ICIR_4h': icir_4h})
        print(f'  {sym}: ICIR_1h={icir_1h:.3f}, ICIR_4h={icir_4h:.3f} '
              f'{"← 4h 更穩" if icir_4h > icir_1h else "← 1h 更穩"}')

    # ══════════════════════════════════════════════════════════
    # 5. Correlation Analysis
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  5. Correlation — 1h Raw / 1h+HTF / 4h Pure')
    print('='*80)

    corr_results = []
    for sym in SYMBOLS:
        df_1h_sym = data_1h[sym]
        df_4h_sym = data_4h[sym]

        sig_1h_raw = make_1h_signal(df_1h_sym['close'], sym)
        sig_1h_htf = simulate_htf_filter(sig_1h_raw, df_1h_sym)
        sig_4h_pure = make_4h_signal(df_4h_sym['close'], sym)

        # Strategy returns (signal_delay=1 via .shift(1))
        ret_1h = df_1h_sym['close'].pct_change().shift(-1)
        strat_A = (sig_1h_raw.shift(1) * ret_1h).fillna(0)
        strat_B = (sig_1h_htf.shift(1) * ret_1h).fillna(0)

        ret_4h = df_4h_sym['close'].pct_change().shift(-1)
        strat_C = (sig_4h_pure.shift(1) * ret_4h).fillna(0)

        daily_A = strat_A.resample('1D').sum()
        daily_B = strat_B.resample('1D').sum()
        daily_C = strat_C.resample('1D').sum()

        common = daily_A.index.intersection(daily_B.index).intersection(daily_C.index)
        a, b, c = daily_A.reindex(common), daily_B.reindex(common), daily_C.reindex(common)

        sr_a = a.mean() / a.std() * np.sqrt(365) if a.std() > 0 else 0
        sr_b = b.mean() / b.std() * np.sqrt(365) if b.std() > 0 else 0
        sr_c = c.mean() / c.std() * np.sqrt(365) if c.std() > 0 else 0

        corr_results.append({
            'symbol': sym,
            'SR_1h_raw': sr_a, 'SR_1h_htf': sr_b, 'SR_4h_pure': sr_c,
            'corr(1h_raw,4h)': a.corr(c),
            'corr(1h_htf,4h)': b.corr(c),
        })

    df_corr = pd.DataFrame(corr_results)
    print(df_corr.round(3).to_string(index=False))

    avg_corr_bc = df_corr['corr(1h_htf,4h)'].mean()
    avg_sr_4h = df_corr['SR_4h_pure'].mean()
    avg_sr_1h_htf = df_corr['SR_1h_htf'].mean()
    print(f'\nAvg corr(1h+HTF, 4h_pure) = {avg_corr_bc:.3f}')
    print(f'Avg SR: 1h+HTF={avg_sr_1h_htf:.3f}, 4h_pure={avg_sr_4h:.3f}')
    if avg_corr_bc < 0.3:
        print('→ ✅ Standalone 價值（corr < 0.3）')
    elif avg_corr_bc < 0.5:
        print('→ ⚠️ 部分獨立，可考慮 blend/overlay')
    else:
        print('→ ❌ 高度冗餘，standalone 價值低')

    # ══════════════════════════════════════════════════════════
    # 6. Turnover & Cost
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  6. Turnover & Cost — 1h vs 4h TSMOM')
    print('='*80)

    to_results = []
    for sym in SYMBOLS:
        sig_1h = make_1h_signal(data_1h[sym]['close'], sym)
        sig_4h = make_4h_signal(data_4h[sym]['close'], sym)

        m1 = compute_turnover_metrics(sig_1h, 8760)
        m4 = compute_turnover_metrics(sig_4h, 2190)

        ratio = m4['turnover_annual'] / m1['turnover_annual'] if m1['turnover_annual'] > 0 else 0
        saving = m1['annual_cost_pct'] - m4['annual_cost_pct']

        to_results.append({
            'symbol': sym,
            'TO_1h': m1['turnover_annual'], 'TO_4h': m4['turnover_annual'],
            '4h/1h': ratio,
            'dir_ch_1h': m1['dir_changes_annual'], 'dir_ch_4h': m4['dir_changes_annual'],
            'hold_1h_h': m1['avg_hold_bars'],
            'hold_4h_h': m4['avg_hold_bars'] * 4,
            'cost_1h%': m1['annual_cost_pct'], 'cost_4h%': m4['annual_cost_pct'],
            'Δcost_pp': saving,
        })

    df_to = pd.DataFrame(to_results)
    print(df_to.round(2).to_string(index=False))

    avg_ratio = df_to['4h/1h'].mean()
    avg_save = df_to['Δcost_pp'].mean()
    print(f'\nAvg turnover ratio (4h/1h): {avg_ratio:.2f}x')
    print(f'Avg cost saving: {avg_save:.2f}pp/yr')

    # ══════════════════════════════════════════════════════════
    # 7. Lookback Sensitivity Sweep
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  7. 4h Lookback Sensitivity (IC on 24h fwd return)')
    print('='*80)

    LOOKBACKS_STD = [21, 42, 63, 84, 126, 168, 252, 336]
    LOOKBACKS_BTC = [90, 180, 270, 360, 504, 720]

    for sym in SYMBOLS:
        df_4h_sym = data_4h[sym]
        fwd = df_4h_sym['close'].pct_change(6).shift(-6)
        lbs = LOOKBACKS_BTC if sym == 'BTCUSDT' else LOOKBACKS_STD

        print(f'\n  {sym}:')
        best_ic, best_lb = -999, 0
        for lb in lbs:
            ef = max(3, lb // 8)
            es = max(8, lb // 3)
            sig = compute_tsmom_signal(
                df_4h_sym['close'], lookback=lb, ema_fast=ef, ema_slow=es,
                annualize_factor=np.sqrt(2190),
            )
            ic = compute_ic(sig, fwd)
            star = ' ← best' if ic == ic and ic > best_ic else ''
            if not np.isnan(ic) and ic > best_ic:
                best_ic, best_lb = ic, lb
                star = ' ← best'
            else:
                star = ''
            print(f'    lb={lb:>4} ({lb*4/24:>5.0f}d) EMA {ef}/{es}: IC={ic:.4f}{star}')
        print(f'    → Best: {best_lb} bars ({best_lb*4/24:.0f}d), IC={best_ic:.4f}')

    # ══════════════════════════════════════════════════════════
    # 8. FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('  FINAL SUMMARY — 4h TSMOM TF Optimization EDA')
    print('='*80)

    summary = []
    for sym in SYMBOLS:
        row_1h = df_ic[(df_ic['symbol'] == sym) & (df_ic['config'] == '1h_prod')]
        rows_4h = df_ic[(df_ic['symbol'] == sym) & (df_ic['config'] != '1h_prod')]
        ic_1h = row_1h['IC_24h_fwd'].values[0]
        best_4h = rows_4h.loc[rows_4h['IC_24h_fwd'].idxmax()]

        corr_row = df_corr[df_corr['symbol'] == sym]
        corr_val = corr_row['corr(1h_htf,4h)'].values[0]
        sr_1h_htf = corr_row['SR_1h_htf'].values[0]
        sr_4h = corr_row['SR_4h_pure'].values[0]

        to_row_i = df_to[df_to['symbol'] == sym]
        cost_save = to_row_i['Δcost_pp'].values[0]

        sub_y = df_yearly[df_yearly['symbol'] == sym]
        icir_4h = sub_y['IC_4h'].mean() / sub_y['IC_4h'].std() if sub_y['IC_4h'].std() > 0 else 0

        summary.append({
            'Symbol': sym,
            'IC_1h': ic_1h,
            'IC_4h': best_4h['IC_24h_fwd'],
            'ΔIC': best_4h['IC_24h_fwd'] - ic_1h,
            'SR_1h_htf': sr_1h_htf,
            'SR_4h': sr_4h,
            'corr': corr_val,
            'ICIR_4h': icir_4h,
            'Δcost_pp': cost_save,
            'best_cfg': best_4h['config'],
        })

    df_sum = pd.DataFrame(summary)
    print(df_sum.round(4).to_string(index=False))

    # Aggregates
    print('\n' + '-'*80)
    m = df_sum.mean(numeric_only=True)
    ic_better = (df_sum['ΔIC'] > 0).sum()
    sr_better = (df_sum['SR_4h'] > df_sum['SR_1h_htf']).sum()

    print(f'Avg IC:          1h={m["IC_1h"]:.4f}  4h={m["IC_4h"]:.4f}  Δ={m["ΔIC"]:+.4f}')
    print(f'4h IC > 1h IC:   {ic_better}/8 symbols')
    print(f'Avg Gross SR:    1h+HTF={m["SR_1h_htf"]:.3f}  4h={m["SR_4h"]:.3f}')
    print(f'4h SR > 1h SR:   {sr_better}/8 symbols')
    print(f'Avg Corr(prod,4h): {m["corr"]:.3f}')
    print(f'Avg ICIR_4h:     {m["ICIR_4h"]:.3f}')
    print(f'Avg Cost Saving: {m["Δcost_pp"]:.2f}pp/yr')

    # ── VERDICT ──
    print('\n' + '='*80)
    print('  VERDICT')
    print('='*80)

    if m['ΔIC'] > 0.005 and m['SR_4h'] > m['SR_1h_htf']:
        print('🟢 4h TSMOM IC 和 SR 均明顯優於 1h → 強烈考慮替換主 TF')
    elif m['ΔIC'] > 0 and m['corr'] > 0.5:
        print(f'🟡 4h IC 略優 (Δ={m["ΔIC"]:+.4f}) 但高度相關 (corr={m["corr"]:.3f})')
        print('   → 不適合作為 standalone leg，但若成本優勢顯著可考慮替換')
        if m['Δcost_pp'] > 0.5:
            print(f'   → 成本節省 {m["Δcost_pp"]:.2f}pp/yr，值得做正式回測')
    elif m['corr'] < 0.3:
        print(f'🟢 corr={m["corr"]:.3f} < 0.3 → 4h TSMOM 有 standalone leg 價值')
    elif m['corr'] < 0.5 and m['SR_4h'] > 0.5:
        print(f'🟡 部分獨立 (corr={m["corr"]:.3f}) + 正 SR ({m["SR_4h"]:.3f})')
        print('   → 考慮作為 blend 組件或替換主 TF（需正式回測）')
    else:
        print(f'🔴 4h 未展現足夠優勢 (ΔIC={m["ΔIC"]:+.4f}, corr={m["corr"]:.3f})')
        print('   → 目前 1h + 4h HTF Filter 組合已接近最優')

    print('\n  [分析注意事項]')
    print('  • 上述 SR 為 gross（未計入 funding rate、slippage），僅供比較用')
    print('  • IC 計算基於 24h 前向收益，與生產策略的 signal_delay=1 一致')
    print('  • 4h EMA 參數為等比例縮減（1h 的 1/4），可能非最優——見 Section 7 sensitivity')
    print('  • 若要進入正式實作，需 Quant Developer 做完整回測（含 cost model + WFA）')


if __name__ == '__main__':
    main()
