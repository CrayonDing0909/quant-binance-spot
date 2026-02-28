"""
鏈上數據探索工具 (Phase 0D)

從免費 API 下載鏈上 (on-chain) 數據，供 Alpha Researcher 探索用：
    1. DeFi Llama — TVL (Total Value Locked)、Stablecoin 流動性
    2. CryptoQuant (free tier) — Exchange Reserve、Whale Alert (需 API key)
    3. Glassnode (free tier) — BTC/ETH 基礎鏈上指標 (需 API key)

這些數據主要作為 Regime Indicator（風險偏好、宏觀環境），
不適合高頻信號（延遲 1-10 分鐘 ~ 數小時）。

儲存路徑：
    data/onchain/{provider}/{metric}.parquet

使用範例：
    # 下載 DeFi Llama 數據（免費，無需 API key）
    PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama

    # 下載特定鏈的 TVL
    PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama --chains ethereum solana bsc

    # 下載 Stablecoin 流動性數據
    PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama --stablecoins

    # 查看已下載數據覆蓋率
    PYTHONPATH=src python scripts/fetch_onchain_data.py --coverage
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/onchain")


# ══════════════════════════════════════════════════════════════
#  DeFi Llama (免費, 無需 API key)
# ══════════════════════════════════════════════════════════════

def fetch_defillama_chain_tvl(chain: str = "Ethereum") -> pd.Series:
    """
    從 DeFi Llama 下載特定鏈的 TVL 歷史

    API: https://api.llama.fi/v2/historicalChainTvl/{chain}
    免費，無 rate limit (合理使用)

    Returns:
        pd.Series indexed by UTC date, values = TVL (USD)
    """
    import requests

    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    logger.info(f"📥 DeFi Llama TVL: {chain}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ DeFi Llama {chain}: {e}")
        return pd.Series(dtype=float, name=f"tvl_{chain.lower()}")

    if not data:
        return pd.Series(dtype=float, name=f"tvl_{chain.lower()}")

    rows = []
    for record in data:
        ts = record.get("date")
        tvl = record.get("tvl", 0)
        if ts is not None:
            rows.append({
                "timestamp": pd.Timestamp(ts, unit="s", tz="UTC"),
                "tvl": float(tvl),
            })

    df = pd.DataFrame(rows)
    series = df.set_index("timestamp")["tvl"].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = f"tvl_{chain.lower()}"

    if not series.empty:
        logger.info(
            f"✅ DeFi Llama TVL {chain}: {len(series)} days "
            f"({series.index[0]:%Y-%m-%d} → {series.index[-1]:%Y-%m-%d})"
        )
    return series


def fetch_defillama_total_tvl() -> pd.Series:
    """
    全鏈 TVL 歷史

    API: https://api.llama.fi/v2/historicalChainTvl
    """
    import requests

    url = "https://api.llama.fi/v2/historicalChainTvl"
    logger.info("📥 DeFi Llama Total TVL")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ DeFi Llama total TVL: {e}")
        return pd.Series(dtype=float, name="tvl_total")

    rows = []
    for record in data:
        ts = record.get("date")
        tvl = record.get("tvl", 0)
        if ts is not None:
            rows.append({
                "timestamp": pd.Timestamp(ts, unit="s", tz="UTC"),
                "tvl": float(tvl),
            })

    df = pd.DataFrame(rows)
    series = df.set_index("timestamp")["tvl"].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = "tvl_total"

    if not series.empty:
        logger.info(
            f"✅ Total TVL: {len(series)} days "
            f"({series.index[0]:%Y-%m-%d} → {series.index[-1]:%Y-%m-%d})"
        )
    return series


def fetch_defillama_stablecoins() -> pd.DataFrame:
    """
    Stablecoin 市值和流動性歷史

    API: https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1
    (stablecoin=1 = USDT, 2 = USDC, etc.)

    Returns:
        DataFrame indexed by date with columns for each stablecoin's market cap
    """
    import requests

    logger.info("📥 DeFi Llama Stablecoin Data")

    # 先取得 stablecoin 列表
    try:
        resp = requests.get("https://stablecoins.llama.fi/stablecoins?includePrices=false", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ DeFi Llama stablecoins list: {e}")
        return pd.DataFrame()

    stablecoins = data.get("peggedAssets", [])
    # 只取前 5 大 stablecoin
    top_stables = sorted(
        stablecoins,
        key=lambda x: x.get("circulating", {}).get("peggedUSD", 0),
        reverse=True,
    )[:5]

    all_series = {}
    for stable in top_stables:
        sc_id = stable.get("id")
        sc_name = stable.get("symbol", f"stable_{sc_id}")

        try:
            url = f"https://stablecoins.llama.fi/stablecoincharts/all?stablecoin={sc_id}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            chart_data = resp.json()
        except Exception as e:
            logger.warning(f"  ⚠️  {sc_name}: {e}")
            continue

        if not chart_data:
            continue

        rows = []
        for record in chart_data:
            ts = record.get("date")
            mcap = record.get("totalCirculating", {}).get("peggedUSD", 0)
            if ts is not None:
                rows.append({
                    "timestamp": pd.Timestamp(int(ts), unit="s", tz="UTC"),
                    f"mcap_{sc_name}": float(mcap),
                })

        if rows:
            s_df = pd.DataFrame(rows).set_index("timestamp")
            series = s_df.iloc[:, 0].sort_index()
            series = series[~series.index.duplicated(keep="last")]
            all_series[sc_name] = series
            logger.info(f"  ✅ {sc_name}: {len(series)} days")

        time.sleep(0.5)  # 禮貌延遲

    if not all_series:
        return pd.DataFrame()

    result = pd.DataFrame(all_series)
    result = result.sort_index()
    return result


def fetch_defillama_protocol_tvl(protocol: str = "aave") -> pd.Series:
    """
    特定協議的 TVL 歷史

    API: https://api.llama.fi/protocol/{protocol}
    """
    import requests

    url = f"https://api.llama.fi/protocol/{protocol}"
    logger.info(f"📥 DeFi Llama Protocol TVL: {protocol}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ DeFi Llama {protocol}: {e}")
        return pd.Series(dtype=float, name=f"tvl_{protocol}")

    tvl_data = data.get("tvl", [])
    rows = []
    for record in tvl_data:
        ts = record.get("date")
        tvl = record.get("totalLiquidityUSD", 0)
        if ts is not None:
            rows.append({
                "timestamp": pd.Timestamp(ts, unit="s", tz="UTC"),
                "tvl": float(tvl),
            })

    if not rows:
        return pd.Series(dtype=float, name=f"tvl_{protocol}")

    df = pd.DataFrame(rows)
    series = df.set_index("timestamp")["tvl"].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = f"tvl_{protocol}"

    if not series.empty:
        logger.info(
            f"✅ {protocol} TVL: {len(series)} days "
            f"({series.index[0]:%Y-%m-%d} → {series.index[-1]:%Y-%m-%d})"
        )
    return series


# ══════════════════════════════════════════════════════════════
#  DeFi Llama Yields (for carry research)
# ══════════════════════════════════════════════════════════════

def fetch_defillama_yields() -> pd.DataFrame:
    """
    下載 DeFi 收益率快照（最新）

    API: https://yields.llama.fi/pools
    可用於 carry 策略研究：DeFi yields vs funding rate

    Returns:
        DataFrame with pool info + APY
    """
    import requests

    url = "https://yields.llama.fi/pools"
    logger.info("📥 DeFi Llama Yields")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ DeFi Llama yields: {e}")
        return pd.DataFrame()

    pools = data.get("data", [])
    if not pools:
        return pd.DataFrame()

    df = pd.DataFrame(pools)
    # 只保留有意義的欄位
    cols = ["chain", "project", "symbol", "tvlUsd", "apy", "apyBase", "apyReward",
            "stablecoin", "exposure", "pool"]
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]

    logger.info(f"✅ DeFi yields: {len(df)} pools")
    return df


# ══════════════════════════════════════════════════════════════
#  儲存 / 載入
# ══════════════════════════════════════════════════════════════

def save_onchain(
    data: pd.Series | pd.DataFrame,
    provider: str,
    metric: str,
    data_dir: Path = DATA_DIR,
) -> Path:
    """儲存鏈上數據到 parquet"""
    path = data_dir / provider / f"{metric}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.to_parquet(path, index=True)
    logger.info(f"💾 Saved {provider}/{metric}: {len(data)} rows → {path}")
    return path


def load_onchain(
    provider: str,
    metric: str,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame | None:
    """載入鏈上數據"""
    path = data_dir / provider / f"{metric}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"⚠️  Load {provider}/{metric} failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  覆蓋率報告
# ══════════════════════════════════════════════════════════════

def show_coverage(data_dir: Path = DATA_DIR) -> None:
    """顯示所有已下載的鏈上數據"""
    if not data_dir.exists():
        print("❌ 無已下載的鏈上數據")
        return

    print("\n📊 鏈上數據覆蓋率報告")
    print("=" * 70)

    for provider_dir in sorted(data_dir.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        print(f"\n  📂 {provider}/")

        for f in sorted(provider_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                metric = f.stem
                if df.empty:
                    print(f"    {metric}: ❌ 空數據")
                else:
                    n = len(df)
                    if hasattr(df.index, 'min'):
                        start = df.index.min()
                        end = df.index.max()
                        print(
                            f"    {metric:<30} ✅ {n:>6} rows  "
                            f"{start} → {end}"
                        )
                    else:
                        print(f"    {metric:<30} ✅ {n:>6} rows")
            except Exception as e:
                print(f"    {f.stem}: ❌ 讀取失敗 ({e})")

    print()


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="鏈上數據探索工具（DeFi Llama / CryptoQuant / Glassnode）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 下載 DeFi Llama TVL 數據
  PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama

  # 下載特定鏈的 TVL
  PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama --chains ethereum solana bsc

  # 下載 Stablecoin 數據
  PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama --stablecoins

  # 下載 DeFi Yields 快照
  PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama --yields

  # 下載特定協議 TVL
  PYTHONPATH=src python scripts/fetch_onchain_data.py --source defillama --protocols aave lido

  # 查看覆蓋率
  PYTHONPATH=src python scripts/fetch_onchain_data.py --coverage
        """,
    )
    parser.add_argument(
        "--source", default="defillama",
        choices=["defillama"],
        help="數據來源 (目前支援: defillama)",
    )
    parser.add_argument(
        "--chains", nargs="+", default=None,
        help="要下載 TVL 的鏈（預設: Ethereum Solana BSC Arbitrum Polygon）",
    )
    parser.add_argument(
        "--protocols", nargs="+", default=None,
        help="要下載 TVL 的協議（e.g. aave lido uniswap）",
    )
    parser.add_argument(
        "--stablecoins", action="store_true",
        help="下載 Top 5 Stablecoin 市值歷史",
    )
    parser.add_argument(
        "--yields", action="store_true",
        help="下載 DeFi 收益率快照（最新）",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="下載所有可用數據",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="只顯示覆蓋率報告",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help=f"數據儲存目錄 (預設: {DATA_DIR})",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.coverage:
        show_coverage(data_dir)
        return

    if args.source == "defillama":
        default_chains = ["Ethereum", "Solana", "BSC", "Arbitrum", "Polygon"]
        chains = args.chains or default_chains

        download_all = args.all

        # 1. Total TVL
        if download_all or (not args.stablecoins and not args.yields and not args.protocols):
            total_tvl = fetch_defillama_total_tvl()
            if not total_tvl.empty:
                save_onchain(total_tvl, "defillama", "tvl_total", data_dir)

        # 2. Per-chain TVL
        if download_all or (not args.stablecoins and not args.yields and not args.protocols):
            for chain in chains:
                series = fetch_defillama_chain_tvl(chain)
                if not series.empty:
                    save_onchain(series, "defillama", f"tvl_{chain.lower()}", data_dir)
                time.sleep(0.3)

        # 3. Protocol TVL
        if args.protocols or download_all:
            protocols = args.protocols or ["aave", "lido", "uniswap", "makerdao", "curve-dex"]
            for protocol in protocols:
                series = fetch_defillama_protocol_tvl(protocol)
                if not series.empty:
                    save_onchain(series, "defillama", f"tvl_{protocol}", data_dir)
                time.sleep(0.3)

        # 4. Stablecoins
        if args.stablecoins or download_all:
            sc_df = fetch_defillama_stablecoins()
            if not sc_df.empty:
                save_onchain(sc_df, "defillama", "stablecoin_mcap", data_dir)

        # 5. Yields
        if args.yields or download_all:
            yields_df = fetch_defillama_yields()
            if not yields_df.empty:
                save_onchain(yields_df, "defillama", "yields_snapshot", data_dir)

    print(f"\n✅ 完成！數據目錄: {data_dir}")


if __name__ == "__main__":
    main()
