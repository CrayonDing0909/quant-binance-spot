from __future__ import annotations
from .broker_interface import BrokerInterface, Order


class BinanceSpotBroker(BrokerInterface):
    """
    Placeholder for future live trading implementation.
    Will wrap Binance REST/WS and maintain account state.
    """

    def get_positions(self) -> dict:
        raise NotImplementedError

    def place_order(self, order: Order) -> dict:
        raise NotImplementedError
