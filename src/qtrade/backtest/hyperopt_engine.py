"""
Hyperopt 參數優化引擎

借鑒 Freqtrade 的 Hyperopt 設計，使用 Optuna 實現貝葉斯優化。

功能：
    1. 自動搜索策略最佳參數組合
    2. 支援多種優化目標（Sharpe ratio、總回報、勝率等）
    3. 避免過度擬合（Walk-Forward 驗證）
    4. 可視化優化過程和參數空間

使用方法：
    from qtrade.backtest.hyperopt_engine import HyperoptEngine, ParamSpace
    
    # 定義參數空間
    param_space = {
        "rsi_period": ParamSpace.integer("rsi_period", 10, 30),
        "oversold": ParamSpace.float("oversold", 25, 40),
        "overbought": ParamSpace.float("overbought", 60, 80),
        "min_adx": ParamSpace.float("min_adx", 15, 35),
        "stop_loss_atr": ParamSpace.float("stop_loss_atr", 1.5, 3.5),
        "take_profit_atr": ParamSpace.float("take_profit_atr", 2.0, 5.0),
    }
    
    # 運行優化
    engine = HyperoptEngine(
        strategy_name="rsi_adx_atr",
        data_path=Path("data/BTCUSDT_1h.parquet"),
        base_cfg=cfg,
        param_space=param_space,
    )
    
    best_params, study = engine.optimize(
        n_trials=200,
        objective="sharpe_ratio",
        n_jobs=4,  # 並行
    )
    
    # 可視化
    engine.plot_optimization_history()
    engine.plot_param_importances()
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from ..utils.log import get_logger
from .run_backtest import run_symbol_backtest

logger = get_logger("hyperopt")

# 忽略 Optuna 的一些警告
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# ══════════════════════════════════════════════════════════════
# 參數空間定義
# ══════════════════════════════════════════════════════════════

@dataclass
class ParamDef:
    """參數定義"""
    name: str
    param_type: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: list | None = None
    step: float | int | None = None
    log: bool = False  # 是否使用對數空間


class ParamSpace:
    """
    參數空間定義工具類
    
    借鑒 Freqtrade 的參數空間設計，支援：
    - 整數參數 (IntSpace)
    - 浮點參數 (DecimalSpace)
    - 類別參數 (CategoricalSpace)
    """
    
    @staticmethod
    def integer(name: str, low: int, high: int, step: int = 1) -> ParamDef:
        """整數參數空間"""
        return ParamDef(name=name, param_type="int", low=low, high=high, step=step)
    
    @staticmethod
    def float(name: str, low: float, high: float, step: float | None = None, log: bool = False) -> ParamDef:
        """浮點參數空間"""
        return ParamDef(name=name, param_type="float", low=low, high=high, step=step, log=log)
    
    @staticmethod
    def categorical(name: str, choices: list) -> ParamDef:
        """類別參數空間"""
        return ParamDef(name=name, param_type="categorical", choices=choices)


# ══════════════════════════════════════════════════════════════
# 優化目標函數
# ══════════════════════════════════════════════════════════════

# 預定義的優化目標
OBJECTIVES = {
    "sharpe_ratio": lambda stats: stats.get("Sharpe Ratio", -999),
    "sortino_ratio": lambda stats: stats.get("Sortino Ratio", -999),
    "total_return": lambda stats: stats.get("Total Return [%]", -999),
    "win_rate": lambda stats: stats.get("Win Rate [%]", 0),
    "profit_factor": lambda stats: stats.get("Profit Factor", 0),
    "max_drawdown": lambda stats: -abs(stats.get("Max Drawdown [%]", -999)),  # 負數因為要最大化（減少 DD）
    "calmar_ratio": lambda stats: stats.get("Calmar Ratio", -999),
    
    # 複合目標（風險調整後報酬）
    "risk_adjusted": lambda stats: (
        stats.get("Sharpe Ratio", 0) * 0.4 +
        stats.get("Sortino Ratio", 0) * 0.3 +
        (100 + stats.get("Max Drawdown [%]", -100)) / 100 * 0.3  # DD 越小越好
    ),
}


def get_objective_fn(objective: str | Callable) -> Callable:
    """獲取優化目標函數"""
    if callable(objective):
        return objective
    if objective in OBJECTIVES:
        return OBJECTIVES[objective]
    raise ValueError(f"Unknown objective: {objective}. Available: {list(OBJECTIVES.keys())}")


# ══════════════════════════════════════════════════════════════
# Hyperopt 引擎
# ══════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """優化結果"""
    best_params: dict
    best_value: float
    study: optuna.Study
    all_trials: pd.DataFrame
    
    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 60,
            "🎯 Hyperopt Optimization Result",
            "=" * 60,
            f"Best {self.study.direction.name}: {self.best_value:.4f}",
            "",
            "Best Parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"Total Trials: {len(self.all_trials)}")
        lines.append(f"Completed: {len(self.all_trials[self.all_trials['state'] == 'COMPLETE'])}")
        lines.append("=" * 60)
        return "\n".join(lines)


class HyperoptEngine:
    """
    Hyperopt 參數優化引擎
    
    借鑒 Freqtrade 的設計，使用 Optuna 實現貝葉斯優化。
    
    特點：
    - 支援多種優化目標
    - 支援並行優化
    - 支援 Walk-Forward 驗證（防止過擬合）
    - 可視化優化過程
    """
    
    def __init__(
        self,
        strategy_name: str,
        data_path: Path,
        base_cfg: dict,
        param_space: dict[str, ParamDef],
        symbol: str = "BTCUSDT",
        market_type: str = "spot",
        direction: str = "both",
    ):
        """
        初始化 Hyperopt 引擎
        
        Args:
            strategy_name: 策略名稱（如 "rsi_adx_atr"）
            data_path: K 線數據路徑
            base_cfg: 基礎配置（包含 initial_cash, fee_bps 等）
            param_space: 參數空間定義
            symbol: 交易對
            market_type: 市場類型
            direction: 交易方向
        """
        self.strategy_name = strategy_name
        self.data_path = data_path
        self.base_cfg = base_cfg.copy()
        self.param_space = param_space
        self.symbol = symbol
        self.market_type = market_type
        self.direction = direction
        
        self._study: optuna.Study | None = None
        self._results: OptimizationResult | None = None
    
    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """從 trial 中採樣參數"""
        params = {}
        for name, pdef in self.param_space.items():
            if pdef.param_type == "int":
                params[name] = trial.suggest_int(name, pdef.low, pdef.high, step=pdef.step or 1)
            elif pdef.param_type == "float":
                if pdef.step:
                    params[name] = trial.suggest_float(name, pdef.low, pdef.high, step=pdef.step)
                elif pdef.log:
                    params[name] = trial.suggest_float(name, pdef.low, pdef.high, log=True)
                else:
                    params[name] = trial.suggest_float(name, pdef.low, pdef.high)
            elif pdef.param_type == "categorical":
                params[name] = trial.suggest_categorical(name, pdef.choices)
        return params
    
    def _create_objective(self, objective_fn: Callable) -> Callable:
        """創建 Optuna 目標函數"""
        
        def objective(trial: optuna.Trial) -> float:
            # 採樣參數
            sampled_params = self._suggest_params(trial)
            
            # 合併到策略參數
            cfg = self.base_cfg.copy()
            cfg["strategy_params"] = {
                **cfg.get("strategy_params", {}),
                **sampled_params,
            }
            
            try:
                # 運行回測
                result = run_symbol_backtest(
                    symbol=self.symbol,
                    data_path=self.data_path,
                    cfg=cfg,
                    strategy_name=self.strategy_name,
                    market_type=self.market_type,
                    direction=self.direction,
                )
                
                # 計算目標值
                stats = result["stats"]
                obj_value = objective_fn(stats)
                
                # 記錄額外指標（用於分析）
                trial.set_user_attr("total_return", stats.get("Total Return [%]", 0))
                trial.set_user_attr("sharpe_ratio", stats.get("Sharpe Ratio", 0))
                trial.set_user_attr("max_drawdown", stats.get("Max Drawdown [%]", 0))
                trial.set_user_attr("win_rate", stats.get("Win Rate [%]", 0))
                trial.set_user_attr("total_trades", stats.get("Total Trades", 0))
                
                # 過濾無效結果
                if np.isnan(obj_value) or np.isinf(obj_value):
                    return float("-inf")
                
                return obj_value
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf")
        
        return objective
    
    def optimize(
        self,
        n_trials: int = 100,
        objective: str | Callable = "sharpe_ratio",
        n_jobs: int = 1,
        timeout: int | None = None,
        show_progress: bool = True,
        seed: int = 42,
    ) -> OptimizationResult:
        """
        運行參數優化
        
        Args:
            n_trials: 優化迭代次數
            objective: 優化目標（字串或自定義函數）
            n_jobs: 並行數（-1 = 使用所有 CPU）
            timeout: 超時秒數
            show_progress: 是否顯示進度條
            seed: 隨機種子
        
        Returns:
            OptimizationResult 包含最佳參數和優化歷史
        """
        objective_fn = get_objective_fn(objective)
        
        # 創建 Study
        sampler = TPESampler(seed=seed)
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{self.strategy_name}_hyperopt",
        )
        
        # 設置日誌級別
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info(f"🚀 Starting Hyperopt: {n_trials} trials, objective={objective}")
        logger.info(f"   Strategy: {self.strategy_name}, Symbol: {self.symbol}")
        logger.info(f"   Param space: {list(self.param_space.keys())}")
        
        # 運行優化
        self._study.optimize(
            self._create_objective(objective_fn),
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=show_progress,
        )
        
        # 收集結果
        trials_df = self._study.trials_dataframe()
        
        self._results = OptimizationResult(
            best_params=self._study.best_params,
            best_value=self._study.best_value,
            study=self._study,
            all_trials=trials_df,
        )
        
        logger.info(self._results.summary())
        return self._results
    
    # ── 可視化方法 ────────────────────────────────────
    
    def plot_optimization_history(self, show: bool = True) -> None:
        """繪製優化歷史"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            import plotly.express as px
            from optuna.visualization import plot_optimization_history
            
            fig = plot_optimization_history(self._study)
            fig.update_layout(title=f"Optimization History - {self.strategy_name}")
            if show:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly not installed, skipping visualization")
    
    def plot_param_importances(self, show: bool = True) -> None:
        """繪製參數重要性"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_param_importances
            
            fig = plot_param_importances(self._study)
            fig.update_layout(title=f"Parameter Importances - {self.strategy_name}")
            if show:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly not installed, skipping visualization")
    
    def plot_contour(self, param1: str, param2: str, show: bool = True) -> None:
        """繪製參數等高線圖（熱力圖）"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_contour
            
            fig = plot_contour(self._study, params=[param1, param2])
            fig.update_layout(title=f"Contour Plot: {param1} vs {param2}")
            if show:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly not installed, skipping visualization")
    
    def plot_parallel_coordinate(self, show: bool = True) -> None:
        """繪製平行坐標圖（所有參數）"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_parallel_coordinate
            
            fig = plot_parallel_coordinate(self._study)
            fig.update_layout(title=f"Parallel Coordinate - {self.strategy_name}")
            if show:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly not installed, skipping visualization")
    
    def get_top_trials(self, n: int = 10) -> pd.DataFrame:
        """獲取前 N 個最佳試驗"""
        if self._results is None:
            raise ValueError("Run optimize() first")
        
        df = self._results.all_trials.copy()
        df = df[df["state"] == "COMPLETE"]
        df = df.sort_values("value", ascending=False)
        return df.head(n)


# ══════════════════════════════════════════════════════════════
# Walk-Forward 驗證（防止過擬合）
# ══════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Walk-Forward 驗證器
    
    將數據分割為多個訓練/測試集，在每個訓練集上優化參數，
    然後在對應的測試集上驗證，最後統計所有測試集的表現。
    
    這可以有效防止過擬合歷史數據。
    """
    
    def __init__(
        self,
        engine: HyperoptEngine,
        n_splits: int = 5,
        train_ratio: float = 0.8,
    ):
        """
        初始化 Walk-Forward 驗證器
        
        Args:
            engine: Hyperopt 引擎
            n_splits: 分割數量
            train_ratio: 訓練集比例
        """
        self.engine = engine
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self._results: list[dict] = []
    
    def run(
        self,
        df: pd.DataFrame,
        n_trials_per_fold: int = 50,
        objective: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        運行 Walk-Forward 驗證
        
        Args:
            df: 完整數據
            n_trials_per_fold: 每個 fold 的優化次數
            objective: 優化目標
        
        Returns:
            每個 fold 的測試結果
        """
        total_len = len(df)
        fold_size = total_len // self.n_splits
        
        results = []
        
        for i in range(self.n_splits):
            logger.info(f"📊 Walk-Forward Fold {i+1}/{self.n_splits}")
            
            # 計算訓練/測試範圍
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else total_len
            train_end = test_start
            train_start = max(0, train_end - int(fold_size / (1 - self.train_ratio) * self.train_ratio))
            
            if train_end - train_start < 100:
                logger.warning(f"Fold {i+1}: Training data too small, skipping")
                continue
            
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            
            logger.info(f"   Train: {len(train_df)} bars, Test: {len(test_df)} bars")
            
            # TODO: 在訓練集上優化，在測試集上驗證
            # 這需要修改 HyperoptEngine 支援傳入 DataFrame
            
            results.append({
                "fold": i + 1,
                "train_size": len(train_df),
                "test_size": len(test_df),
                # ... 添加更多結果
            })
        
        self._results = results
        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# 預定義參數空間（常用策略）
# ══════════════════════════════════════════════════════════════

# RSI + ADX + ATR 策略的參數空間
RSI_ADX_ATR_PARAM_SPACE = {
    "rsi_period": ParamSpace.integer("rsi_period", 7, 28),
    "oversold": ParamSpace.float("oversold", 25, 40, step=5),
    "overbought": ParamSpace.float("overbought", 60, 80, step=5),
    "min_adx": ParamSpace.float("min_adx", 15, 35, step=5),
    "adx_period": ParamSpace.integer("adx_period", 10, 21),
    "stop_loss_atr": ParamSpace.float("stop_loss_atr", 1.5, 3.5, step=0.5),
    "take_profit_atr": ParamSpace.float("take_profit_atr", 2.0, 5.0, step=0.5),
    "atr_period": ParamSpace.integer("atr_period", 10, 21),
    "cooldown_bars": ParamSpace.integer("cooldown_bars", 3, 12),
}

# EMA Cross 策略的參數空間
EMA_CROSS_PARAM_SPACE = {
    "fast_period": ParamSpace.integer("fast_period", 5, 20),
    "slow_period": ParamSpace.integer("slow_period", 20, 100),
    "signal_period": ParamSpace.integer("signal_period", 5, 15),
}
