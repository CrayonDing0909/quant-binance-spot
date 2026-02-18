"""
Hyperopt 參數優化引擎

借鑒 Freqtrade 的 Hyperopt 設計，使用 Optuna 實現貝葉斯優化。

功能：
    1. 自動搜索策略最佳參數組合（TPE / CMA-ES / Grid）
    2. 支援多種優化目標（Sharpe ratio、總回報、勝率等）
    3. 多幣種聯合優化（跨資產魯棒性）
    4. Train/Test OOS 驗證（防止過擬合）
    5. Walk-Forward 驗證（滾動 OOS 驗證）
    6. 可視化優化過程和參數空間

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

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler, CmaEsSampler, GridSampler

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
    
    # Sharpe - DD 懲罰（最實用的目標函數）
    "sharpe_dd": lambda stats: (
        stats.get("Sharpe Ratio", 0) -
        0.02 * abs(stats.get("Max Drawdown [%]", 50))  # DD 太大要扣分
    ),
}


def get_objective_fn(objective: str | Callable) -> Callable:
    """獲取優化目標函數"""
    if callable(objective):
        return objective
    if objective in OBJECTIVES:
        return OBJECTIVES[objective]
    raise ValueError(f"Unknown objective: {objective}. Available: {list(OBJECTIVES.keys())}")


def _min_trades_penalty(stats: dict, min_trades: int = 10) -> float:
    """交易次數過少的懲罰（防止過擬合到少數完美交易）"""
    total_trades = stats.get("Total Trades", 0)
    if total_trades < min_trades:
        return -100.0  # 嚴重懲罰
    return 0.0


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
    # OOS 結果（如果有）
    oos_stats: dict | None = None
    
    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 60,
            "🎯 Hyperopt Optimization Result",
            "=" * 60,
            f"Best objective value: {self.best_value:.4f}",
            "",
            "Best Parameters:",
        ]
        for k, v in self.best_params.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        
        completed = len(self.all_trials[self.all_trials['state'] == 'COMPLETE']) if 'state' in self.all_trials.columns else len(self.all_trials)
        lines.append("")
        lines.append(f"Total Trials: {len(self.all_trials)}")
        lines.append(f"Completed: {completed}")
        
        if self.oos_stats:
            lines.append("")
            lines.append("─── Out-of-Sample (OOS) Results ───")
            for k, v in self.oos_stats.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """導出為可序列化的 dict"""
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.all_trials),
            "oos_stats": self.oos_stats,
        }


class HyperoptEngine:
    """
    Hyperopt 參數優化引擎
    
    借鑒 Freqtrade 的設計，使用 Optuna 實現貝葉斯優化。
    
    特點：
    - 支援多種優化目標
    - 支援並行優化
    - 支援 TPE / CMA-ES / Grid 多種搜索算法
    - 多幣種聯合優化
    - Train/Test OOS 驗證（防止過擬合）
    - 可視化優化過程
    """
    
    def __init__(
        self,
        strategy_name: str,
        symbol_data: dict[str, Path],
        base_cfg: dict,
        param_space: dict[str, ParamDef],
        market_type: str = "spot",
        direction: str = "both",
        data_dir: Path | None = None,
        min_trades: int = 10,
    ):
        """
        初始化 Hyperopt 引擎
        
        Args:
            strategy_name: 策略名稱（如 "rsi_adx_atr"）
            symbol_data: {symbol: data_path} 字典，支援多幣種
            base_cfg: 基礎配置（包含 initial_cash, fee_bps 等）
            param_space: 參數空間定義
            market_type: 市場類型
            direction: 交易方向
            data_dir: 數據根目錄（用於 funding rate 等）
            min_trades: 最低交易次數（防止過擬合）
        """
        self.strategy_name = strategy_name
        self.symbol_data = symbol_data  # {symbol: Path}
        self.base_cfg = base_cfg.copy()
        self.param_space = param_space
        self.market_type = market_type
        self.direction = direction
        self.data_dir = data_dir
        self.min_trades = min_trades
        
        self._study: optuna.Study | None = None
        self._results: OptimizationResult | None = None
    
    # 向後相容的便捷建構子
    @classmethod
    def from_single_symbol(
        cls,
        strategy_name: str,
        data_path: Path,
        base_cfg: dict,
        param_space: dict[str, ParamDef],
        symbol: str = "BTCUSDT",
        **kwargs,
    ) -> "HyperoptEngine":
        """從單幣種建立引擎（向後相容）"""
        return cls(
            strategy_name=strategy_name,
            symbol_data={symbol: data_path},
            base_cfg=base_cfg,
            param_space=param_space,
            **kwargs,
        )
    
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
    
    def _run_backtest_for_symbol(
        self, symbol: str, data_path: Path, cfg: dict,
    ) -> dict:
        """對單一幣種運行回測，返回 stats dict"""
        result = run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=cfg,
            strategy_name=self.strategy_name,
            market_type=self.market_type,
            direction=self.direction,
            data_dir=self.data_dir,
        )
        return result.stats
    
    def _create_objective(self, objective_fn: Callable) -> Callable:
        """創建 Optuna 目標函數（支援多幣種聯合優化）"""
        
        def objective(trial: optuna.Trial) -> float:
            # 採樣參數
            sampled_params = self._suggest_params(trial)
            
            # 合併到策略參數
            cfg = self.base_cfg.copy()
            cfg["strategy_params"] = {
                **cfg.get("strategy_params", {}),
                **sampled_params,
            }
            
            # 多幣種聯合評估
            all_obj_values = []
            all_stats = {}
            
            for symbol, data_path in self.symbol_data.items():
                try:
                    stats = self._run_backtest_for_symbol(symbol, data_path, cfg)
                    
                    # 最低交易次數懲罰
                    penalty = _min_trades_penalty(stats, self.min_trades)
                    if penalty < 0:
                        trial.set_user_attr(f"{symbol}_penalty", "low_trades")
                        all_obj_values.append(penalty)
                        continue
                    
                    obj_value = objective_fn(stats)
                    
                    if np.isnan(obj_value) or np.isinf(obj_value):
                        all_obj_values.append(-999.0)
                        continue
                    
                    all_obj_values.append(obj_value)
                    all_stats[symbol] = {
                        "total_return": stats.get("Total Return [%]", 0),
                        "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                        "max_drawdown": stats.get("Max Drawdown [%]", 0),
                        "win_rate": stats.get("Win Rate [%]", 0),
                        "total_trades": stats.get("Total Trades", 0),
                    }
                    
                except Exception as e:
                    logger.warning(f"Trial {trial.number} - {symbol} failed: {e}")
                    all_obj_values.append(-999.0)
            
            if not all_obj_values:
                return float("-inf")
            
            # 聯合目標：取平均（跨幣種魯棒性）
            combined = float(np.mean(all_obj_values))
            
            # 記錄到 trial attributes
            for symbol, s in all_stats.items():
                for k, v in s.items():
                    trial.set_user_attr(f"{symbol}_{k}", v)
            trial.set_user_attr("combined_objective", combined)
            trial.set_user_attr("n_symbols", len(self.symbol_data))
            
            # 如果單幣種，也記錄主要指標到頂層
            if len(all_stats) == 1:
                s = list(all_stats.values())[0]
                trial.set_user_attr("total_return", s["total_return"])
                trial.set_user_attr("sharpe_ratio", s["sharpe_ratio"])
                trial.set_user_attr("max_drawdown", s["max_drawdown"])
                trial.set_user_attr("win_rate", s["win_rate"])
                trial.set_user_attr("total_trades", s["total_trades"])
            elif all_stats:
                # 多幣種：記錄平均值
                for k in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"]:
                    avg = np.mean([s[k] for s in all_stats.values()])
                    trial.set_user_attr(f"avg_{k}", avg)
            
            return combined
        
        return objective
    
    def _create_sampler(
        self,
        method: str = "tpe",
        seed: int = 42,
    ) -> optuna.samplers.BaseSampler:
        """根據搜索方法創建對應的 sampler"""
        if method == "tpe":
            return TPESampler(seed=seed, multivariate=True)
        elif method == "cmaes":
            return CmaEsSampler(seed=seed)
        elif method == "grid":
            # Grid sampler 需要搜索空間定義
            search_space = {}
            for name, pdef in self.param_space.items():
                if pdef.param_type == "int":
                    step = pdef.step or 1
                    search_space[name] = list(range(pdef.low, pdef.high + 1, step))
                elif pdef.param_type == "float":
                    if pdef.step:
                        vals = []
                        v = pdef.low
                        while v <= pdef.high + 1e-9:
                            vals.append(round(v, 6))
                            v += pdef.step
                        search_space[name] = vals
                    else:
                        # 無 step 的 float → 自動離散化為 5 個點
                        search_space[name] = [
                            round(pdef.low + (pdef.high - pdef.low) * i / 4, 4)
                            for i in range(5)
                        ]
                elif pdef.param_type == "categorical":
                    search_space[name] = pdef.choices
            return GridSampler(search_space, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method}. Available: tpe, cmaes, grid")
    
    def optimize(
        self,
        n_trials: int = 100,
        objective: str | Callable = "sharpe_ratio",
        method: str = "tpe",
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
            method: 搜索算法 ("tpe", "cmaes", "grid")
            n_jobs: 並行數（-1 = 使用所有 CPU）
            timeout: 超時秒數
            show_progress: 是否顯示進度條
            seed: 隨機種子
        
        Returns:
            OptimizationResult 包含最佳參數和優化歷史
        """
        objective_fn = get_objective_fn(objective)
        objective_name = objective if isinstance(objective, str) else "custom"
        
        # 創建 Sampler
        sampler = self._create_sampler(method, seed)
        
        # 創建 Study
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{self.strategy_name}_hyperopt",
        )
        
        # 設置日誌級別
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        
        symbols_str = ", ".join(self.symbol_data.keys())
        logger.info(f"🚀 Starting Hyperopt: {n_trials} trials, method={method}, objective={objective_name}")
        logger.info(f"   Strategy: {self.strategy_name}, Symbols: [{symbols_str}]")
        logger.info(f"   Param space ({len(self.param_space)} params): {list(self.param_space.keys())}")
        
        t0 = time.time()
        
        # 運行優化
        self._study.optimize(
            self._create_objective(objective_fn),
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=show_progress,
        )
        
        elapsed = time.time() - t0
        logger.info(f"⏱️  Optimization completed in {elapsed:.1f}s ({elapsed/max(n_trials,1):.1f}s/trial)")
        
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
    
    def run_oos_validation(
        self,
        best_params: dict,
        oos_data: dict[str, Path],
        objective_fn: Callable | str = "sharpe_ratio",
    ) -> dict:
        """
        用最佳參數在 OOS（Out-of-Sample）數據上驗證
        
        Args:
            best_params: 最佳參數
            oos_data: {symbol: oos_data_path} OOS 數據
            objective_fn: 目標函數（用於計算 OOS objective）
        
        Returns:
            OOS 結果 dict
        """
        if isinstance(objective_fn, str):
            objective_fn = get_objective_fn(objective_fn)
        
        cfg = self.base_cfg.copy()
        cfg["strategy_params"] = {
            **cfg.get("strategy_params", {}),
            **best_params,
        }
        
        oos_results = {}
        obj_values = []
        
        for symbol, data_path in oos_data.items():
            try:
                stats = self._run_backtest_for_symbol(symbol, data_path, cfg)
                oos_results[symbol] = {
                    "total_return": stats.get("Total Return [%]", 0),
                    "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                    "max_drawdown": stats.get("Max Drawdown [%]", 0),
                    "win_rate": stats.get("Win Rate [%]", 0),
                    "total_trades": stats.get("Total Trades", 0),
                }
                obj_values.append(objective_fn(stats))
            except Exception as e:
                logger.warning(f"OOS validation failed for {symbol}: {e}")
                oos_results[symbol] = {"error": str(e)}
        
        # 匯總
        oos_summary = {
            "per_symbol": oos_results,
            "avg_objective": float(np.mean(obj_values)) if obj_values else None,
            "n_symbols": len(oos_data),
        }
        
        if self._results:
            self._results.oos_stats = oos_summary
        
        return oos_summary
    
    # ── 可視化方法 ────────────────────────────────────
    
    def plot_optimization_history(self, save_path: Path | None = None) -> None:
        """繪製優化歷史"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_optimization_history
            
            fig = plot_optimization_history(self._study)
            fig.update_layout(title=f"Optimization History - {self.strategy_name}")
            if save_path:
                fig.write_image(str(save_path))
                logger.info(f"📊 Saved optimization history: {save_path}")
            else:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly/kaleido not installed, skipping visualization")
    
    def plot_param_importances(self, save_path: Path | None = None) -> None:
        """繪製參數重要性"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_param_importances
            
            fig = plot_param_importances(self._study)
            fig.update_layout(title=f"Parameter Importances - {self.strategy_name}")
            if save_path:
                fig.write_image(str(save_path))
                logger.info(f"📊 Saved param importances: {save_path}")
            else:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly/kaleido not installed, skipping visualization")
    
    def plot_contour(self, param1: str, param2: str, save_path: Path | None = None) -> None:
        """繪製參數等高線圖（熱力圖）"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_contour
            
            fig = plot_contour(self._study, params=[param1, param2])
            fig.update_layout(title=f"Contour Plot: {param1} vs {param2}")
            if save_path:
                fig.write_image(str(save_path))
            else:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly/kaleido not installed, skipping visualization")
    
    def plot_parallel_coordinate(self, save_path: Path | None = None) -> None:
        """繪製平行坐標圖（所有參數）"""
        if self._study is None:
            raise ValueError("Run optimize() first")
        
        try:
            from optuna.visualization import plot_parallel_coordinate
            
            fig = plot_parallel_coordinate(self._study)
            fig.update_layout(title=f"Parallel Coordinate - {self.strategy_name}")
            if save_path:
                fig.write_image(str(save_path))
            else:
                fig.show()
            return fig
        except ImportError:
            logger.warning("plotly/kaleido not installed, skipping visualization")
    
    def get_top_trials(self, n: int = 10) -> pd.DataFrame:
        """獲取前 N 個最佳試驗"""
        if self._results is None:
            raise ValueError("Run optimize() first")
        
        df = self._results.all_trials.copy()
        if "state" in df.columns:
            df = df[df["state"] == "COMPLETE"]
        df = df.sort_values("value", ascending=False)
        return df.head(n)
    
    def save_results(self, output_dir: Path) -> None:
        """將結果儲存到目錄"""
        if self._results is None:
            raise ValueError("Run optimize() first")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 最佳參數 JSON
        result_dict = self._results.to_dict()
        with open(output_dir / "best_params.json", "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # 2. 所有 trial 的 CSV
        self._results.all_trials.to_csv(output_dir / "all_trials.csv", index=False)
        
        # 3. Top 20 結果
        top = self.get_top_trials(20)
        top.to_csv(output_dir / "top_trials.csv", index=False)
        
        # 4. 摘要文字
        with open(output_dir / "summary.txt", "w") as f:
            f.write(self._results.summary())
        
        logger.info(f"💾 Results saved to: {output_dir}")


# ══════════════════════════════════════════════════════════════
# Walk-Forward 驗證（防止過擬合）
# ══════════════════════════════════════════════════════════════

@dataclass
class WalkForwardFold:
    """單一 fold 的結果"""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    train_objective: float
    test_objective: float
    test_stats: dict


class WalkForwardValidator:
    """
    Walk-Forward 驗證器
    
    將數據分割為多個訓練/測試集，在每個訓練集上優化參數，
    然後在對應的測試集上驗證，最後統計所有測試集的表現。
    
    這可以有效防止過擬合歷史數據。
    
    使用 Anchored Walk-Forward：
    - 訓練集從數據起點開始，逐漸增長
    - 測試集固定大小，逐步往前推移
    """
    
    def __init__(
        self,
        strategy_name: str,
        symbol_data: dict[str, Path],
        base_cfg: dict,
        param_space: dict[str, ParamDef],
        market_type: str = "spot",
        direction: str = "both",
        data_dir: Path | None = None,
    ):
        self.strategy_name = strategy_name
        self.symbol_data = symbol_data
        self.base_cfg = base_cfg
        self.param_space = param_space
        self.market_type = market_type
        self.direction = direction
        self.data_dir = data_dir
        self._folds: list[WalkForwardFold] = []
    
    def run(
        self,
        n_splits: int = 5,
        n_trials_per_fold: int = 50,
        objective: str = "sharpe_ratio",
        method: str = "tpe",
        min_train_bars: int = 2000,
    ) -> pd.DataFrame:
        """
        運行 Anchored Walk-Forward 驗證
        
        數據被分為 n_splits+1 段：
        - Fold 1: train=[0, 1], test=[2]
        - Fold 2: train=[0, 1, 2], test=[3]
        - ...
        - Fold n: train=[0..n], test=[n+1]
        
        Args:
            n_splits: 測試 fold 數量
            n_trials_per_fold: 每個 fold 的優化試驗數
            objective: 優化目標
            method: 搜索算法
            min_train_bars: 最少訓練數據 bar 數
        
        Returns:
            DataFrame with per-fold results
        """
        from ..data.storage import load_klines
        
        objective_fn = get_objective_fn(objective)
        self._folds = []
        
        # 先載入數據確定完整長度（用第一個幣種的數據長度）
        first_symbol = list(self.symbol_data.keys())[0]
        first_path = self.symbol_data[first_symbol]
        full_df = load_klines(first_path)
        total_len = len(full_df)
        
        # 計算每個 segment 的大小
        n_segments = n_splits + 1
        segment_size = total_len // n_segments
        
        if segment_size < 500:
            logger.warning(f"每個 segment 只有 {segment_size} bars，可能不足以可靠優化")
        
        logger.info(f"🔄 Walk-Forward: {n_splits} folds, {total_len} total bars")
        logger.info(f"   Segment size: ~{segment_size} bars")
        
        for fold_i in range(n_splits):
            train_end_idx = (fold_i + 1) * segment_size
            test_start_idx = train_end_idx
            test_end_idx = min((fold_i + 2) * segment_size, total_len)
            
            if train_end_idx < min_train_bars:
                logger.warning(f"Fold {fold_i+1}: Training data ({train_end_idx} bars) < min_train_bars ({min_train_bars}), skipping")
                continue
            
            train_dates = (
                full_df.index[0].strftime("%Y-%m-%d"),
                full_df.index[train_end_idx - 1].strftime("%Y-%m-%d"),
            )
            test_dates = (
                full_df.index[test_start_idx].strftime("%Y-%m-%d"),
                full_df.index[test_end_idx - 1].strftime("%Y-%m-%d"),
            )
            
            logger.info(f"\n📊 Fold {fold_i+1}/{n_splits}")
            logger.info(f"   Train: {train_dates[0]} → {train_dates[1]} ({train_end_idx} bars)")
            logger.info(f"   Test:  {test_dates[0]} → {test_dates[1]} ({test_end_idx - test_start_idx} bars)")
            
            # 訓練：在 train 數據上建立回測配置（修改 start/end）
            train_cfg = self.base_cfg.copy()
            train_cfg["start"] = train_dates[0]
            train_cfg["end"] = train_dates[1]
            
            engine = HyperoptEngine(
                strategy_name=self.strategy_name,
                symbol_data=self.symbol_data,
                base_cfg=train_cfg,
                param_space=self.param_space,
                market_type=self.market_type,
                direction=self.direction,
                data_dir=self.data_dir,
            )
            
            result = engine.optimize(
                n_trials=n_trials_per_fold,
                objective=objective,
                method=method,
                show_progress=False,
            )
            
            # 測試：用最佳參數在 test 數據上回測
            test_cfg = self.base_cfg.copy()
            test_cfg["start"] = test_dates[0]
            test_cfg["end"] = test_dates[1]
            test_cfg["strategy_params"] = {
                **test_cfg.get("strategy_params", {}),
                **result.best_params,
            }
            
            test_stats_all = {}
            test_obj_values = []
            
            for symbol, data_path in self.symbol_data.items():
                try:
                    test_result = run_symbol_backtest(
                        symbol=symbol,
                        data_path=data_path,
                        cfg=test_cfg,
                        strategy_name=self.strategy_name,
                        market_type=self.market_type,
                        direction=self.direction,
                        data_dir=self.data_dir,
                    )
                    stats = test_result.stats
                    test_stats_all[symbol] = {
                        "total_return": stats.get("Total Return [%]", 0),
                        "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                        "max_drawdown": stats.get("Max Drawdown [%]", 0),
                        "total_trades": stats.get("Total Trades", 0),
                    }
                    test_obj_values.append(objective_fn(stats))
                except Exception as e:
                    logger.warning(f"Fold {fold_i+1} test failed for {symbol}: {e}")
            
            test_obj = float(np.mean(test_obj_values)) if test_obj_values else -999
            
            fold = WalkForwardFold(
                fold_idx=fold_i + 1,
                train_start=train_dates[0],
                train_end=train_dates[1],
                test_start=test_dates[0],
                test_end=test_dates[1],
                best_params=result.best_params,
                train_objective=result.best_value,
                test_objective=test_obj,
                test_stats=test_stats_all,
            )
            self._folds.append(fold)
            
            logger.info(f"   Train obj: {result.best_value:.4f} → Test obj: {test_obj:.4f}")
        
        # 生成結果 DataFrame
        rows = []
        for f in self._folds:
            row = {
                "fold": f.fold_idx,
                "train_period": f"{f.train_start} → {f.train_end}",
                "test_period": f"{f.test_start} → {f.test_end}",
                "train_objective": f.train_objective,
                "test_objective": f.test_objective,
                "overfit_ratio": (f.train_objective / f.test_objective) if f.test_objective != 0 else float("inf"),
            }
            # 加入最佳參數
            for k, v in f.best_params.items():
                row[f"param_{k}"] = v
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            avg_train = df["train_objective"].mean()
            avg_test = df["test_objective"].mean()
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Walk-Forward Summary")
            logger.info(f"   Avg Train Objective: {avg_train:.4f}")
            logger.info(f"   Avg Test Objective:  {avg_test:.4f}")
            logger.info(f"   Overfit Ratio:       {avg_train/avg_test:.2f}x" if avg_test != 0 else "   Overfit Ratio: N/A")
            logger.info(f"   Test > 0 folds:      {(df['test_objective'] > 0).sum()}/{len(df)}")
            logger.info(f"{'='*60}")
        
        return df


# ══════════════════════════════════════════════════════════════
# Train/Test OOS 分割工具
# ══════════════════════════════════════════════════════════════

def split_data_for_oos(
    data_path: Path,
    train_ratio: float = 0.7,
) -> tuple[Path, Path]:
    """
    將數據按時間分割為 Train/Test，寫入臨時檔案
    
    Args:
        data_path: 原始數據路徑
        train_ratio: 訓練集比例
    
    Returns:
        (train_path, test_path)
    """
    from ..data.storage import load_klines
    
    df = load_klines(data_path)
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # 儲存到臨時位置
    parent = data_path.parent
    stem = data_path.stem
    suffix = data_path.suffix
    
    train_path = parent / f"{stem}_train{suffix}"
    test_path = parent / f"{stem}_test{suffix}"
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    logger.info(f"📂 Split data: Train {len(train_df)} bars ({train_df.index[0].strftime('%Y-%m-%d')} → {train_df.index[-1].strftime('%Y-%m-%d')})")
    logger.info(f"              Test  {len(test_df)} bars ({test_df.index[0].strftime('%Y-%m-%d')} → {test_df.index[-1].strftime('%Y-%m-%d')})")
    
    return train_path, test_path


def cleanup_oos_files(data_paths: dict[str, Path]) -> None:
    """清理 OOS 分割產生的臨時檔案"""
    for symbol, path in data_paths.items():
        for suffix in ["_train", "_test"]:
            tmp = path.parent / f"{path.stem}{suffix}{path.suffix}"
            if tmp.exists():
                tmp.unlink()
                logger.debug(f"Cleaned up: {tmp}")


# ══════════════════════════════════════════════════════════════
# 預定義參數空間（常用策略）
# ══════════════════════════════════════════════════════════════

# RSI + ADX + ATR 策略 — 核心參數
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

# RSI + ADX + ATR — 擴展參數（含 Dynamic RSI + Adaptive SL + HTF）
RSI_ADX_ATR_EXTENDED_PARAM_SPACE = {
    # 核心
    "rsi_period": ParamSpace.integer("rsi_period", 7, 24),
    "min_adx": ParamSpace.float("min_adx", 12, 30, step=3),
    "short_min_adx": ParamSpace.float("short_min_adx", 15, 35, step=5),
    "adx_period": ParamSpace.integer("adx_period", 10, 21),
    "stop_loss_atr": ParamSpace.float("stop_loss_atr", 1.5, 3.0, step=0.5),
    "atr_period": ParamSpace.integer("atr_period", 10, 21),
    "cooldown_bars": ParamSpace.integer("cooldown_bars", 2, 8),
    "min_hold_bars": ParamSpace.integer("min_hold_bars", 1, 4),
    
    # Dynamic RSI
    "rsi_lookback_days": ParamSpace.integer("rsi_lookback_days", 7, 28),
    "rsi_quantile_low": ParamSpace.float("rsi_quantile_low", 0.05, 0.20, step=0.05),
    "rsi_quantile_high": ParamSpace.float("rsi_quantile_high", 0.80, 0.95, step=0.05),
    
    # Adaptive SL
    "er_sl_min": ParamSpace.float("er_sl_min", 1.0, 2.0, step=0.25),
    "er_sl_max": ParamSpace.float("er_sl_max", 2.0, 3.5, step=0.25),
    "adaptive_sl_er_period": ParamSpace.integer("adaptive_sl_er_period", 8, 14),
    
    # HTF soft weights
    "htf_ema_fast": ParamSpace.integer("htf_ema_fast", 10, 30),
    "htf_ema_slow": ParamSpace.integer("htf_ema_slow", 30, 70),
    "htf_counter_weight": ParamSpace.float("htf_counter_weight", 0.3, 0.7, step=0.1),
    "htf_neutral_weight": ParamSpace.float("htf_neutral_weight", 0.5, 0.9, step=0.1),
    
    # 波動率過濾
    "min_atr_ratio": ParamSpace.float("min_atr_ratio", 0.003, 0.010, step=0.001),
    
    # 波動率 Regime 倉位縮放
    "vol_regime_low_pct": ParamSpace.float("vol_regime_low_pct", 20, 50, step=5),
    "vol_regime_low_weight": ParamSpace.float("vol_regime_low_weight", 0.3, 0.7, step=0.1),
    "vol_regime_lookback": ParamSpace.integer("vol_regime_lookback", 96, 336),  # 4天~14天
}

# EMA Cross 策略的參數空間（預留）
EMA_CROSS_PARAM_SPACE = {
    "fast_period": ParamSpace.integer("fast_period", 5, 20),
    "slow_period": ParamSpace.integer("slow_period", 20, 100),
    "signal_period": ParamSpace.integer("signal_period", 5, 15),
}

# TSMOM 策略 — 核心參數
TSMOM_PARAM_SPACE = {
    "lookback": ParamSpace.integer("lookback", 48, 720),
    "vol_target": ParamSpace.float("vol_target", 0.05, 0.30, step=0.05),
}

# TSMOM + EMA — 擴展參數
TSMOM_EMA_PARAM_SPACE = {
    "lookback": ParamSpace.integer("lookback", 48, 720),
    "vol_target": ParamSpace.float("vol_target", 0.05, 0.30, step=0.05),
    "ema_fast": ParamSpace.integer("ema_fast", 10, 30),
    "ema_slow": ParamSpace.integer("ema_slow", 30, 100),
    "agree_weight": ParamSpace.float("agree_weight", 0.8, 1.2, step=0.1),
    "disagree_weight": ParamSpace.float("disagree_weight", 0.1, 0.5, step=0.1),
}

# TSMOM Multi + EMA — 全參數空間
TSMOM_MULTI_EMA_PARAM_SPACE = {
    "vol_target": ParamSpace.float("vol_target", 0.05, 0.30, step=0.05),
    "ema_fast": ParamSpace.integer("ema_fast", 10, 30),
    "ema_slow": ParamSpace.integer("ema_slow", 30, 100),
    "agree_weight": ParamSpace.float("agree_weight", 0.8, 1.2, step=0.1),
    "disagree_weight": ParamSpace.float("disagree_weight", 0.1, 0.5, step=0.1),
    "vol_regime_enabled": ParamSpace.categorical("vol_regime_enabled", [True, False]),
    "vol_regime_lookback": ParamSpace.integer("vol_regime_lookback", 100, 300),
    "vol_regime_low_pct": ParamSpace.float("vol_regime_low_pct", 20.0, 40.0, step=5.0),
    "vol_regime_low_weight": ParamSpace.float("vol_regime_low_weight", 0.3, 0.7, step=0.1),
}

# 預定義空間名稱查找
XSMOM_PARAM_SPACE = {
    "lookbacks": ParamSpace.categorical("lookbacks", [
        "[336,720,1440]", "[336,720]", "[720,1440]", "[336,720,1440,2160]",
    ]),
    "skip_recent": ParamSpace.integer("skip_recent", 0, 72, step=24),
    "use_residual": ParamSpace.categorical("use_residual", [True, False]),
    "long_threshold": ParamSpace.float("long_threshold", 0.6, 0.8, step=0.05),
    "short_threshold": ParamSpace.float("short_threshold", 0.2, 0.4, step=0.05),
    "vol_target": ParamSpace.float("vol_target", 0.10, 0.30, step=0.05),
    "scale_mode": ParamSpace.categorical("scale_mode", ["threshold", "linear"]),
}

XSMOM_TSMOM_PARAM_SPACE = {
    **XSMOM_PARAM_SPACE,
    "xsmom_weight": ParamSpace.float("xsmom_weight", 0.2, 0.5, step=0.1),
    "tsmom_weight": ParamSpace.float("tsmom_weight", 0.5, 0.8, step=0.1),
    "tsmom_lookback": ParamSpace.integer("tsmom_lookback", 72, 336, step=24),
    "ema_fast": ParamSpace.integer("ema_fast", 10, 30),
    "ema_slow": ParamSpace.integer("ema_slow", 40, 80),
}

PREDEFINED_SPACES = {
    "rsi_adx_atr": RSI_ADX_ATR_PARAM_SPACE,
    "rsi_adx_atr_extended": RSI_ADX_ATR_EXTENDED_PARAM_SPACE,
    "ema_cross": EMA_CROSS_PARAM_SPACE,
    "tsmom": TSMOM_PARAM_SPACE,
    "tsmom_ema": TSMOM_EMA_PARAM_SPACE,
    "tsmom_multi_ema": TSMOM_MULTI_EMA_PARAM_SPACE,
    "xsmom": XSMOM_PARAM_SPACE,
    "xsmom_tsmom": XSMOM_TSMOM_PARAM_SPACE,
}


def get_param_space(name: str) -> dict[str, ParamDef]:
    """根據名稱獲取預定義參數空間"""
    if name in PREDEFINED_SPACES:
        return PREDEFINED_SPACES[name]
    raise ValueError(f"Unknown param space: {name}. Available: {list(PREDEFINED_SPACES.keys())}")
