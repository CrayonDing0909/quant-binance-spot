import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_validate_live_consistency_module():
    module_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "validate_live_consistency.py"
    )
    spec = importlib.util.spec_from_file_location("validate_live_consistency", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fee_match_reports_bps_as_percent():
    module = _load_validate_live_consistency_module()
    checker = module.ConsistencyChecker.__new__(module.ConsistencyChecker)
    checker.cfg = SimpleNamespace(backtest=SimpleNamespace(fee_bps=4))
    checker.results = []

    module.ConsistencyChecker.check_fee_match(checker)

    assert checker.results
    assert checker.results[0].message == "fee=4bps = Binance Taker (0.04%)"
