# tests/test_should_stop.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crossfuse_v5 import should_stop


def test_continue_when_nothing_triggers():
    assert should_stop(2, 6, 1, 1000.0, 30000.0) is None


def test_patience_only_applies_in_finetune_phase():
    assert should_stop(9, 6, 0, 10.0, None) is None
    assert should_stop(6, 6, 1, 10.0, None) == "patience"


def test_time_budget_stops_in_any_phase_when_exceeded():
    assert should_stop(0, 6, 1, 30001.0, 30000.0) == "time_budget"
    assert should_stop(0, 6, 0, 30001.0, 30000.0) == "time_budget"


def test_no_budget_means_no_time_stop():
    assert should_stop(0, 6, 1, 10**9, None) is None
