from __future__ import annotations
from dataclasses import dataclass
from .broker_interface import BrokerInterface


@dataclass
class LiveState:
    is_running: bool = False


class LiveRunner:
    """
    Placeholder state machine for live execution.
    """

    def __init__(self, broker: BrokerInterface) -> None:
        self.broker = broker
        self.state = LiveState()

    def start(self) -> None:
        self.state.is_running = True

    def stop(self) -> None:
        self.state.is_running = False
