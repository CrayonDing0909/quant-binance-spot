from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Order:
    symbol: str
    side: str  # BUY/SELL
    qty: float
    type: str = "MARKET"


class BrokerInterface:
    def get_positions(self) -> dict:
        raise NotImplementedError

    def place_order(self, order: Order) -> dict:
        raise NotImplementedError
