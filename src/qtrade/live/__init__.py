from .paper_broker import PaperBroker
from .binance_spot_broker import BinanceSpotBroker
from .binance_futures_broker import BinanceFuturesBroker
from .signal_generator import generate_signal
from .runner import LiveRunner
from .trading_state import TradingStateManager, TradingState, TradeLog, get_state_manager

__all__ = [
    "PaperBroker",
    "BinanceSpotBroker",
    "BinanceFuturesBroker",
    "generate_signal",
    "LiveRunner",
    "TradingStateManager",
    "TradingState",
    "TradeLog",
    "get_state_manager",
]

# TODO: 合約風控（待完成）
#   - 強平價格計算與預警
#   - 資金費率追蹤與統計
#   - 保證金率監控