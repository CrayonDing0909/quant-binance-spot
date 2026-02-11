from .paper_broker import PaperBroker
from .binance_spot_broker import BinanceSpotBroker
from .signal_generator import generate_signal
from .runner import LiveRunner
from .trading_state import TradingStateManager, TradingState, TradeLog, get_state_manager

# TODO: BinanceFuturesBroker 實作
#   - 合約 API 整合 (/fapi/v1/*)
#   - 設置槓桿、保證金模式
#   - 開/平多倉、開/平空倉
#   - 查詢持倉、餘額
#   - 止損止盈掛單

# TODO: 合約風控
#   - 強平價格計算與預警
#   - 資金費率追蹤與統計
#   - 保證金率監控