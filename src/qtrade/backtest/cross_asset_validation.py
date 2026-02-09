"""
Cross-Asset 驗證模組

提供策略在不同資產間的穩健性驗證功能：
- Leave-One-Asset-Out (LOAO) 驗證
- 相關性分層驗證
- 市場狀態分層驗證
- 驗證結果分析

設計原則：
- 單一職責：每個類別/函數只做一件事
- 依賴注入：回測函數可替換，便於測試
- 結果不可變：使用 frozen dataclass
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 型別定義
# ══════════════════════════════════════════════════════════════════════════════

class BacktestFunction(Protocol):
    """回測函數介面"""
    def __call__(
        self,
        symbol: str,
        data_path: Path,
        cfg: dict,
        strategy_name: str | None = None,
    ) -> dict:
        """
        執行單一資產回測
        
        Returns:
            包含 "stats" 的字典
        """
        ...


class DataLoader(Protocol):
    """數據載入介面"""
    def __call__(self, data_path: Path) -> pd.DataFrame:
        """載入 K 線數據"""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# 列舉與常數
# ══════════════════════════════════════════════════════════════════════════════

class ValidationMethod(Enum):
    """驗證方法類型"""
    LEAVE_ONE_OUT = "leave_one_out"
    CORRELATION_STRATIFIED = "correlation_stratified"
    MARKET_REGIME = "market_regime"


class MarketRegimeIndicator(Enum):
    """市場狀態指標"""
    VOLATILITY = "volatility"
    TREND = "trend"
    MOMENTUM = "momentum"


class RobustnessLevel(Enum):
    """穩健性等級"""
    ROBUST = "robust"
    MODERATE = "moderate"
    WEAK = "weak"
    OVERFITTED = "overfitted"


# ══════════════════════════════════════════════════════════════════════════════
# 配置類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CrossAssetValidationConfig:
    """
    Cross-Asset 驗證配置
    
    Attributes:
        min_bars_per_asset: 每個資產最少需要的 K 線數量
        min_assets: 最少需要的資產數量
        parallel: 是否並行執行
        max_workers: 最大並行工作數
        degradation_threshold: 績效衰退警告閾值
        severe_degradation_threshold: 嚴重績效衰退閾值
    """
    min_bars_per_asset: int = 500
    min_assets: int = 3
    parallel: bool = True
    max_workers: int = 4
    degradation_threshold: float = 0.3
    severe_degradation_threshold: float = 0.5


@dataclass(frozen=True)
class CorrelationStratifiedConfig:
    """相關性分層驗證配置"""
    n_groups: int = 3
    min_assets_per_group: int = 1
    clustering_method: str = "ward"  # ward, complete, average, single


@dataclass(frozen=True)
class MarketRegimeConfig:
    """市場狀態驗證配置"""
    indicator: MarketRegimeIndicator = MarketRegimeIndicator.VOLATILITY
    lookback_period: int = 20
    sma_period: int = 50
    min_bars_per_regime: int = 500


# ══════════════════════════════════════════════════════════════════════════════
# 結果類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AssetValidationResult:
    """單一資產驗證結果"""
    asset: str
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_drawdown: float
    test_drawdown: float
    sharpe_degradation: float
    return_degradation: float
    
    @property
    def is_overfitted(self) -> bool:
        """判斷是否過擬合"""
        return self.sharpe_degradation > 0.5 or self.return_degradation > 0.5


@dataclass(frozen=True)
class CrossAssetValidationResult:
    """Cross-Asset 驗證結果"""
    method: ValidationMethod
    asset_results: Tuple[AssetValidationResult, ...]
    avg_train_sharpe: float
    avg_test_sharpe: float
    avg_sharpe_degradation: float
    std_sharpe_degradation: float
    overfitted_assets: Tuple[str, ...]
    robustness_level: RobustnessLevel
    warnings: Tuple[str, ...]
    
    def to_dataframe(self) -> pd.DataFrame:
        """轉換為 DataFrame"""
        return pd.DataFrame([
            {
                "asset": r.asset,
                "train_sharpe": r.train_sharpe,
                "test_sharpe": r.test_sharpe,
                "train_return": r.train_return,
                "test_return": r.test_return,
                "sharpe_degradation": r.sharpe_degradation,
                "is_overfitted": r.is_overfitted,
            }
            for r in self.asset_results
        ])


@dataclass(frozen=True)
class CorrelationGroupResult:
    """相關性分組驗證結果"""
    group_id: int
    train_assets: Tuple[str, ...]
    test_assets: Tuple[str, ...]
    avg_train_sharpe: float
    avg_test_sharpe: float
    avg_correlation_within_group: float


@dataclass(frozen=True)
class MarketRegimeResult:
    """市場狀態驗證結果"""
    symbol: str  # 改為 symbol 以保持一致
    regime: str
    n_bars: int
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float


# ══════════════════════════════════════════════════════════════════════════════
# 驗證器基類
# ══════════════════════════════════════════════════════════════════════════════

class BaseValidator(ABC):
    """
    驗證器基類
    
    定義驗證器的介面和共用方法。
    """
    
    def __init__(
        self,
        backtest_func: BacktestFunction,
        data_loader: DataLoader,
        config: CrossAssetValidationConfig | None = None,
    ):
        """
        Args:
            backtest_func: 回測函數
            data_loader: 數據載入函數
            config: 驗證配置
        """
        self._backtest_func = backtest_func
        self._data_loader = data_loader
        self._config = config or CrossAssetValidationConfig()
    
    @abstractmethod
    def validate(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> CrossAssetValidationResult:
        """
        執行驗證
        
        Args:
            symbols: 資產列表
            data_paths: 資產數據路徑映射
            cfg: 策略配置
        
        Returns:
            驗證結果
        """
        raise NotImplementedError
    
    def _run_backtest_safe(
        self,
        symbol: str,
        data_path: Path,
        cfg: dict,
    ) -> dict | None:
        """
        安全執行回測
        
        Returns:
            回測結果或 None（若失敗）
        """
        try:
            return self._backtest_func(
                symbol,
                data_path,
                cfg,
                cfg.get("strategy_name"),
            )
        except Exception as e:
            warnings.warn(f"回測失敗 {symbol}: {e}")
            return None
    
    def _extract_stats(self, result: dict | None) -> dict:
        """提取統計數據"""
        if result is None:
            return {
                "Sharpe Ratio": 0.0,
                "Total Return [%]": 0.0,
                "Max Drawdown [%]": 0.0,
                "Win Rate [%]": 0.0,
            }
        return result.get("stats", {})
    
    def _calculate_degradation(
        self,
        train_value: float,
        test_value: float,
    ) -> float:
        """計算績效衰退比例"""
        if abs(train_value) < 1e-10:
            return 0.0
        return (train_value - test_value) / abs(train_value)
    
    def _determine_robustness(
        self,
        avg_degradation: float,
        std_degradation: float,
        overfitted_ratio: float,
    ) -> RobustnessLevel:
        """判斷穩健性等級"""
        if overfitted_ratio > 0.5:
            return RobustnessLevel.OVERFITTED
        if avg_degradation > self._config.severe_degradation_threshold:
            return RobustnessLevel.WEAK
        if avg_degradation > self._config.degradation_threshold:
            return RobustnessLevel.MODERATE
        return RobustnessLevel.ROBUST


# ══════════════════════════════════════════════════════════════════════════════
# Leave-One-Asset-Out 驗證器
# ══════════════════════════════════════════════════════════════════════════════

class LeaveOneAssetOutValidator(BaseValidator):
    """
    Leave-One-Asset-Out 驗證器
    
    對每個資產：用其他所有資產的平均績效作為訓練基準，
    檢測策略是否在特定資產上過擬合。
    
    使用範例:
        validator = LeaveOneAssetOutValidator(run_symbol_backtest, load_klines)
        result = validator.validate(symbols, data_paths, cfg)
        print(result.robustness_level)
    """
    
    def validate(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> CrossAssetValidationResult:
        """執行 LOAO 驗證"""
        if len(symbols) < self._config.min_assets:
            raise ValueError(
                f"至少需要 {self._config.min_assets} 個資產，"
                f"目前只有 {len(symbols)} 個"
            )
        
        # 執行驗證
        if self._config.parallel:
            asset_results = self._validate_parallel(symbols, data_paths, cfg)
        else:
            asset_results = self._validate_sequential(symbols, data_paths, cfg)
        
        # 過濾無效結果
        valid_results = [r for r in asset_results if r is not None]
        
        if not valid_results:
            return self._empty_result()
        
        # 計算統計數據
        degradations = [r.sharpe_degradation for r in valid_results]
        overfitted = [r.asset for r in valid_results if r.is_overfitted]
        
        warnings_list = self._generate_warnings(valid_results, degradations)
        
        return CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=tuple(valid_results),
            avg_train_sharpe=np.mean([r.train_sharpe for r in valid_results]),
            avg_test_sharpe=np.mean([r.test_sharpe for r in valid_results]),
            avg_sharpe_degradation=np.mean(degradations),
            std_sharpe_degradation=np.std(degradations),
            overfitted_assets=tuple(overfitted),
            robustness_level=self._determine_robustness(
                np.mean(degradations),
                np.std(degradations),
                len(overfitted) / len(valid_results),
            ),
            warnings=tuple(warnings_list),
        )
    
    def _validate_single_asset(
        self,
        test_symbol: str,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> AssetValidationResult | None:
        """驗證單一資產"""
        train_symbols = [s for s in symbols if s != test_symbol]
        
        # 訓練集回測
        train_stats_list = []
        for symbol in train_symbols:
            if symbol not in data_paths:
                continue
            result = self._run_backtest_safe(symbol, data_paths[symbol], cfg)
            if result:
                train_stats_list.append(self._extract_stats(result))
        
        if not train_stats_list:
            return None
        
        # 測試集回測
        if test_symbol not in data_paths:
            return None
        
        test_result = self._run_backtest_safe(
            test_symbol,
            data_paths[test_symbol],
            cfg,
        )
        if test_result is None:
            return None
        
        test_stats = self._extract_stats(test_result)
        
        # 計算平均訓練績效
        train_sharpe = np.mean([s.get("Sharpe Ratio", 0) for s in train_stats_list])
        train_return = np.mean([s.get("Total Return [%]", 0) for s in train_stats_list])
        train_dd = np.mean([abs(s.get("Max Drawdown [%]", 0)) for s in train_stats_list])
        
        test_sharpe = test_stats.get("Sharpe Ratio", 0)
        test_return = test_stats.get("Total Return [%]", 0)
        test_dd = abs(test_stats.get("Max Drawdown [%]", 0))
        
        return AssetValidationResult(
            asset=test_symbol,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_return=train_return,
            test_return=test_return,
            train_drawdown=train_dd,
            test_drawdown=test_dd,
            sharpe_degradation=self._calculate_degradation(train_sharpe, test_sharpe),
            return_degradation=self._calculate_degradation(train_return, test_return),
        )
    
    def _validate_parallel(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> List[AssetValidationResult | None]:
        """並行驗證"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._validate_single_asset,
                    symbol,
                    symbols,
                    data_paths,
                    cfg,
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                results.append(future.result())
        
        return results
    
    def _validate_sequential(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> List[AssetValidationResult | None]:
        """順序驗證"""
        return [
            self._validate_single_asset(symbol, symbols, data_paths, cfg)
            for symbol in symbols
        ]
    
    def _generate_warnings(
        self,
        results: List[AssetValidationResult],
        degradations: List[float],
    ) -> List[str]:
        """生成警告訊息"""
        warnings_list = []
        
        avg_deg = np.mean(degradations)
        if avg_deg > self._config.degradation_threshold:
            warnings_list.append(
                f"平均績效衰退 ({avg_deg:.1%}) 超過閾值 "
                f"({self._config.degradation_threshold:.0%})"
            )
        
        overfitted = [r.asset for r in results if r.is_overfitted]
        if overfitted:
            warnings_list.append(f"可能過擬合的資產: {overfitted}")
        
        if np.std(degradations) > 0.5:
            warnings_list.append(
                f"績效衰退變異度高 (std={np.std(degradations):.2f})"
            )
        
        return warnings_list
    
    def _empty_result(self) -> CrossAssetValidationResult:
        """返回空結果"""
        return CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=(),
            avg_train_sharpe=0.0,
            avg_test_sharpe=0.0,
            avg_sharpe_degradation=0.0,
            std_sharpe_degradation=0.0,
            overfitted_assets=(),
            robustness_level=RobustnessLevel.WEAK,
            warnings=("無有效驗證結果",),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 相關性分層驗證器
# ══════════════════════════════════════════════════════════════════════════════

class CorrelationStratifiedValidator(BaseValidator):
    """
    相關性分層驗證器
    
    將資產按相關性分組，用高相關性資產訓練，低相關性資產測試，
    檢測策略是否依賴特定的資產相關結構。
    """
    
    def __init__(
        self,
        backtest_func: BacktestFunction,
        data_loader: DataLoader,
        config: CrossAssetValidationConfig | None = None,
        stratified_config: CorrelationStratifiedConfig | None = None,
    ):
        super().__init__(backtest_func, data_loader, config)
        self._stratified_config = stratified_config or CorrelationStratifiedConfig()
    
    def validate(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> CrossAssetValidationResult:
        """執行相關性分層驗證"""
        # 1. 計算相關性矩陣
        corr_matrix = self._compute_correlation_matrix(symbols, data_paths)
        
        if corr_matrix is None or len(corr_matrix) < self._config.min_assets:
            return self._empty_result()
        
        # 2. 層次聚類分組
        groups = self._cluster_assets(corr_matrix)
        
        # 3. 交叉驗證
        group_results = self._cross_validate_groups(groups, data_paths, cfg, corr_matrix)
        
        if not group_results:
            return self._empty_result()
        
        # 4. 轉換為標準結果格式
        asset_results = self._convert_to_asset_results(group_results)
        degradations = [r.sharpe_degradation for r in asset_results]
        overfitted = [r.asset for r in asset_results if r.is_overfitted]
        
        return CrossAssetValidationResult(
            method=ValidationMethod.CORRELATION_STRATIFIED,
            asset_results=tuple(asset_results),
            avg_train_sharpe=np.mean([r.avg_train_sharpe for r in group_results]),
            avg_test_sharpe=np.mean([r.avg_test_sharpe for r in group_results]),
            avg_sharpe_degradation=np.mean(degradations) if degradations else 0.0,
            std_sharpe_degradation=np.std(degradations) if degradations else 0.0,
            overfitted_assets=tuple(overfitted),
            robustness_level=self._determine_robustness(
                np.mean(degradations) if degradations else 0.0,
                np.std(degradations) if degradations else 0.0,
                len(overfitted) / max(len(asset_results), 1),
            ),
            warnings=(),
        )
    
    def _compute_correlation_matrix(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
    ) -> pd.DataFrame | None:
        """計算資產間相關性矩陣"""
        returns_dict = {}
        
        for symbol in symbols:
            if symbol not in data_paths:
                continue
            try:
                df = self._data_loader(data_paths[symbol])
                returns_dict[symbol] = df["close"].pct_change().dropna()
            except Exception:
                continue
        
        if len(returns_dict) < 3:
            return None
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        return returns_df.corr()
    
    def _cluster_assets(
        self,
        corr_matrix: pd.DataFrame,
    ) -> Dict[int, List[str]]:
        """使用層次聚類將資產分組"""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # 轉換為距離矩陣
        dist_matrix = 1 - corr_matrix.abs()
        np.fill_diagonal(dist_matrix.values, 0)
        
        # 層次聚類
        condensed_dist = squareform(dist_matrix.values)
        linkage_matrix = linkage(
            condensed_dist,
            method=self._stratified_config.clustering_method,
        )
        clusters = fcluster(
            linkage_matrix,
            self._stratified_config.n_groups,
            criterion="maxclust",
        )
        
        # 建立分組映射
        groups: Dict[int, List[str]] = {}
        for symbol, cluster_id in zip(corr_matrix.columns, clusters):
            if cluster_id not in groups:
                groups[cluster_id] = []
            groups[cluster_id].append(symbol)
        
        return groups
    
    def _cross_validate_groups(
        self,
        groups: Dict[int, List[str]],
        data_paths: Dict[str, Path],
        cfg: dict,
        corr_matrix: pd.DataFrame,
    ) -> List[CorrelationGroupResult]:
        """對分組進行交叉驗證"""
        results = []
        group_ids = list(groups.keys())
        
        for test_group_id in group_ids:
            train_symbols = []
            for gid in group_ids:
                if gid != test_group_id:
                    train_symbols.extend(groups[gid])
            test_symbols = groups[test_group_id]
            
            # 訓練集回測
            train_stats = self._batch_backtest(train_symbols, data_paths, cfg)
            
            # 測試集回測
            test_stats = self._batch_backtest(test_symbols, data_paths, cfg)
            
            if train_stats and test_stats:
                # 計算組內平均相關性
                if len(test_symbols) > 1:
                    test_corr = corr_matrix.loc[test_symbols, test_symbols]
                    mask = np.triu_indices(len(test_symbols), k=1)
                    avg_corr = test_corr.values[mask].mean()
                else:
                    avg_corr = 0.0
                
                results.append(CorrelationGroupResult(
                    group_id=test_group_id,
                    train_assets=tuple(train_symbols),
                    test_assets=tuple(test_symbols),
                    avg_train_sharpe=np.mean([s.get("Sharpe Ratio", 0) for s in train_stats]),
                    avg_test_sharpe=np.mean([s.get("Sharpe Ratio", 0) for s in test_stats]),
                    avg_correlation_within_group=avg_corr,
                ))
        
        return results
    
    def _batch_backtest(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> List[dict]:
        """批次執行回測"""
        stats_list = []
        for symbol in symbols:
            if symbol not in data_paths:
                continue
            result = self._run_backtest_safe(symbol, data_paths[symbol], cfg)
            if result:
                stats_list.append(self._extract_stats(result))
        return stats_list
    
    def _convert_to_asset_results(
        self,
        group_results: List[CorrelationGroupResult],
    ) -> List[AssetValidationResult]:
        """轉換分組結果為資產結果"""
        asset_results = []
        
        for gr in group_results:
            degradation = self._calculate_degradation(
                gr.avg_train_sharpe,
                gr.avg_test_sharpe,
            )
            
            for asset in gr.test_assets:
                asset_results.append(AssetValidationResult(
                    asset=asset,
                    train_sharpe=gr.avg_train_sharpe,
                    test_sharpe=gr.avg_test_sharpe,
                    train_return=0.0,  # 分組驗證不追蹤個別收益
                    test_return=0.0,
                    train_drawdown=0.0,
                    test_drawdown=0.0,
                    sharpe_degradation=degradation,
                    return_degradation=degradation,
                ))
        
        return asset_results
    
    def _empty_result(self) -> CrossAssetValidationResult:
        """返回空結果"""
        return CrossAssetValidationResult(
            method=ValidationMethod.CORRELATION_STRATIFIED,
            asset_results=(),
            avg_train_sharpe=0.0,
            avg_test_sharpe=0.0,
            avg_sharpe_degradation=0.0,
            std_sharpe_degradation=0.0,
            overfitted_assets=(),
            robustness_level=RobustnessLevel.WEAK,
            warnings=("資產數量不足以進行相關性分層驗證",),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 市場狀態驗證器
# ══════════════════════════════════════════════════════════════════════════════

class MarketRegimeValidator(BaseValidator):
    """
    市場狀態驗證器
    
    將時間區段按市場狀態分類，驗證策略在不同狀態下的表現。
    """
    
    def __init__(
        self,
        backtest_func: BacktestFunction,
        data_loader: DataLoader,
        config: CrossAssetValidationConfig | None = None,
        regime_config: MarketRegimeConfig | None = None,
    ):
        super().__init__(backtest_func, data_loader, config)
        self._regime_config = regime_config or MarketRegimeConfig()
    
    def validate(
        self,
        symbols: List[str],
        data_paths: Dict[str, Path],
        cfg: dict,
    ) -> Tuple[List[MarketRegimeResult], pd.DataFrame]:
        """
        執行市場狀態驗證
        
        Returns:
            (結果列表, 摘要 DataFrame)
        """
        results = []
        
        for symbol in symbols:
            if symbol not in data_paths:
                continue
            
            symbol_results = self._validate_symbol_regimes(
                symbol,
                data_paths[symbol],
                cfg,
            )
            results.extend(symbol_results)
        
        # 建立摘要 DataFrame
        if results:
            summary_df = pd.DataFrame([
                {
                    "symbol": r.symbol,
                    "regime": r.regime,
                    "n_bars": r.n_bars,
                    "sharpe": r.sharpe,
                    "return": r.total_return,
                    "max_dd": r.max_drawdown,
                    "win_rate": r.win_rate,
                }
                for r in results
            ])
        else:
            summary_df = pd.DataFrame()
        
        return results, summary_df
    
    def _validate_symbol_regimes(
        self,
        symbol: str,
        data_path: Path,
        cfg: dict,
    ) -> List[MarketRegimeResult]:
        """驗證單一資產在不同市場狀態下的表現"""
        try:
            df = self._data_loader(data_path)
        except Exception:
            return []
        
        # 計算市場狀態指標
        regime_mask = self._compute_regime_mask(df)
        
        results = []
        for regime_name, mask in [("high", regime_mask), ("low", ~regime_mask)]:
            regime_df = df[mask].copy()
            
            if len(regime_df) < self._regime_config.min_bars_per_regime:
                continue
            
            # 保存臨時數據
            temp_path = data_path.parent / f"_temp_{symbol}_{regime_name}.parquet"
            
            try:
                regime_df.to_parquet(temp_path)
                result = self._run_backtest_safe(symbol, temp_path, cfg)
                
                if result:
                    stats = self._extract_stats(result)
                    results.append(MarketRegimeResult(
                        symbol=symbol,
                        regime=f"{self._regime_config.indicator.value}_{regime_name}",
                        n_bars=len(regime_df),
                        sharpe=stats.get("Sharpe Ratio", 0),
                        total_return=stats.get("Total Return [%]", 0),
                        max_drawdown=stats.get("Max Drawdown [%]", 0),
                        win_rate=stats.get("Win Rate [%]", 0),
                    ))
            finally:
                temp_path.unlink(missing_ok=True)
        
        return results
    
    def _compute_regime_mask(self, df: pd.DataFrame) -> pd.Series:
        """計算市場狀態遮罩"""
        indicator = self._regime_config.indicator
        
        if indicator == MarketRegimeIndicator.VOLATILITY:
            vol = df["close"].pct_change().rolling(
                self._regime_config.lookback_period
            ).std() * np.sqrt(252)
            return vol > vol.median()
        
        elif indicator == MarketRegimeIndicator.TREND:
            sma = df["close"].rolling(self._regime_config.sma_period).mean()
            trend = sma.diff(10) / sma.shift(10)
            return trend.abs() > trend.abs().median()
        
        else:  # MOMENTUM
            momentum = df["close"].pct_change(self._regime_config.lookback_period)
            return momentum > momentum.median()


# ══════════════════════════════════════════════════════════════════════════════
# 結果分析器
# ══════════════════════════════════════════════════════════════════════════════

class ValidationResultAnalyzer:
    """
    驗證結果分析器
    
    提供驗證結果的深度分析和報告生成功能。
    """
    
    @staticmethod
    def summarize(result: CrossAssetValidationResult) -> dict:
        """生成結果摘要"""
        return {
            "method": result.method.value,
            "n_assets": len(result.asset_results),
            "robustness": result.robustness_level.value,
            "avg_train_sharpe": round(result.avg_train_sharpe, 3),
            "avg_test_sharpe": round(result.avg_test_sharpe, 3),
            "avg_degradation": f"{result.avg_sharpe_degradation:.1%}",
            "std_degradation": f"{result.std_sharpe_degradation:.2f}",
            "overfitted_assets": list(result.overfitted_assets),
            "warnings": list(result.warnings),
        }
    
    @staticmethod
    def is_strategy_robust(result: CrossAssetValidationResult) -> bool:
        """判斷策略是否穩健"""
        return result.robustness_level in (
            RobustnessLevel.ROBUST,
            RobustnessLevel.MODERATE,
        )
    
    @staticmethod
    def get_recommendations(result: CrossAssetValidationResult) -> List[str]:
        """根據結果生成改進建議"""
        recommendations = []
        
        if result.robustness_level == RobustnessLevel.OVERFITTED:
            recommendations.append("策略可能嚴重過擬合，建議簡化策略參數或使用正則化")
        
        if result.robustness_level == RobustnessLevel.WEAK:
            recommendations.append("策略跨資產泛化能力弱，建議增加訓練資產多樣性")
        
        if result.overfitted_assets:
            recommendations.append(
                f"建議移除以下資產的特定優化: {list(result.overfitted_assets)}"
            )
        
        if result.std_sharpe_degradation > 0.4:
            recommendations.append("績效衰退變異度高，策略表現不穩定")
        
        if not recommendations:
            recommendations.append("策略跨資產驗證通過，可進行下一步測試")
        
        return recommendations


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函數
# ══════════════════════════════════════════════════════════════════════════════

def leave_one_asset_out(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    backtest_func: BacktestFunction | None = None,
    data_loader: DataLoader | None = None,
    parallel: bool = True,
) -> CrossAssetValidationResult:
    """
    Leave-One-Asset-Out 驗證便捷函數
    
    Args:
        symbols: 資產列表
        data_paths: 資產數據路徑映射
        cfg: 策略配置
        backtest_func: 回測函數（默認使用 run_symbol_backtest）
        data_loader: 數據載入函數（默認使用 load_klines）
        parallel: 是否並行執行
    
    Returns:
        CrossAssetValidationResult
    
    Example:
        result = leave_one_asset_out(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            data_paths={s: Path(f"data/{s}.parquet") for s in symbols},
            cfg=strategy_config,
        )
        print(result.robustness_level)
    """
    if backtest_func is None:
        from .run_backtest import run_symbol_backtest
        backtest_func = run_symbol_backtest
    
    if data_loader is None:
        from ..data.storage import load_klines
        data_loader = load_klines
    
    config = CrossAssetValidationConfig(parallel=parallel)
    validator = LeaveOneAssetOutValidator(backtest_func, data_loader, config)
    
    return validator.validate(symbols, data_paths, cfg)


def correlation_stratified_validation(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    n_groups: int = 3,
    backtest_func: BacktestFunction | None = None,
    data_loader: DataLoader | None = None,
) -> CrossAssetValidationResult:
    """
    相關性分層驗證便捷函數
    
    Args:
        symbols: 資產列表
        data_paths: 資產數據路徑映射
        cfg: 策略配置
        n_groups: 分組數量
        backtest_func: 回測函數
        data_loader: 數據載入函數
    
    Returns:
        CrossAssetValidationResult
    """
    if backtest_func is None:
        from .run_backtest import run_symbol_backtest
        backtest_func = run_symbol_backtest
    
    if data_loader is None:
        from ..data.storage import load_klines
        data_loader = load_klines
    
    stratified_config = CorrelationStratifiedConfig(n_groups=n_groups)
    validator = CorrelationStratifiedValidator(
        backtest_func,
        data_loader,
        stratified_config=stratified_config,
    )
    
    return validator.validate(symbols, data_paths, cfg)


def market_regime_validation(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    indicator: str = "volatility",
    backtest_func: BacktestFunction | None = None,
    data_loader: DataLoader | None = None,
) -> Tuple[List[MarketRegimeResult], pd.DataFrame]:
    """
    市場狀態驗證便捷函數
    
    Args:
        symbols: 資產列表
        data_paths: 資產數據路徑映射
        cfg: 策略配置
        indicator: 市場狀態指標 ("volatility", "trend", "momentum")
        backtest_func: 回測函數
        data_loader: 數據載入函數
    
    Returns:
        (MarketRegimeResult 列表, 摘要 DataFrame)
    """
    if backtest_func is None:
        from .run_backtest import run_symbol_backtest
        backtest_func = run_symbol_backtest
    
    if data_loader is None:
        from ..data.storage import load_klines
        data_loader = load_klines
    
    indicator_enum = MarketRegimeIndicator(indicator)
    regime_config = MarketRegimeConfig(indicator=indicator_enum)
    validator = MarketRegimeValidator(
        backtest_func,
        data_loader,
        regime_config=regime_config,
    )
    
    return validator.validate(symbols, data_paths, cfg)
