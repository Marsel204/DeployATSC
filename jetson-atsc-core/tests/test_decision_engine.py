"""
Unit tests for pure ATSCDecisionEngine.
Verifies control boundary conditions, ANFIS inference isolation, and failsafe execution.
"""

import os
import sys
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.common.types import TrafficSnapshot, ActuationDecision
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine


@pytest.fixture
def decision_engine():
    model_path = os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx")
    anfis = ANFISInferenceEngine(model_path, prefer_cuda=False, min_green_sec=10.0, max_green_sec=120.0)
    return ATSCDecisionEngine(anfis, min_green_sec=10.0, max_green_sec=120.0, fallback_green_sec=30.0)


def test_decision_engine_empty_intersection(decision_engine):
    snap = TrafficSnapshot(v_w_pcu=0.0, queue_meters=0.0, occupancy_pct=0.0)
    dec = decision_engine.evaluate(snap, cycle_id=1)

    assert isinstance(dec, ActuationDecision)
    assert dec.cycle_id == 1
    assert dec.green_seconds >= 10.0
    assert dec.green_seconds <= 120.0
    assert dec.failsafe_active is False
    assert dec.inference_latency_ms > 0.0


def test_decision_engine_high_demand(decision_engine):
    snap = TrafficSnapshot(v_w_pcu=60.0, queue_meters=45.0, occupancy_pct=95.0)
    dec = decision_engine.evaluate(snap, cycle_id=2)

    assert dec.cycle_id == 2
    assert dec.green_seconds > 40.0
    assert dec.green_seconds <= 120.0
    assert dec.failsafe_active is False


def test_decision_engine_fallback():
    class BrokenANFIS:
        def evaluate_green_time(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory simulation")

    engine = ATSCDecisionEngine(BrokenANFIS(), min_green_sec=10.0, max_green_sec=120.0, fallback_green_sec=30.0)
    snap = TrafficSnapshot(v_w_pcu=10.0, queue_meters=10.0, occupancy_pct=20.0)
    dec = engine.evaluate(snap, cycle_id=99)

    assert dec.cycle_id == 99
    assert dec.green_seconds == 30.0
    assert dec.failsafe_active is True


def synthetic_anfis(raw=10.,failure=False):
    import numpy as np
    class Session:
        def run(self,*args):
            if failure:raise RuntimeError('synthetic backend failure')
            return [np.array([[raw]],dtype=np.float64)]
    engine=object.__new__(ANFISInferenceEngine)
    engine.session=Session();engine.input_name='x';engine.output_name='y'
    engine.min_green=10.;engine.max_green=120.;engine.active_provider='SyntheticExecutionProvider'
    return engine


@pytest.mark.parametrize('raw,bounded,rounded,clipped,touch,rounding',[
    (10.,10.,10.,False,True,False),(-5.,10.,10.,True,True,False),
    (10.004,10.004,10.,False,False,True),(121.,120.,120.,True,False,False)])
def test_per_call_raw_bounded_and_rounded_diagnostics(raw,bounded,rounded,clipped,touch,rounding):
    engine=synthetic_anfis(raw)
    decision=ATSCDecisionEngine(engine).evaluate(TrafficSnapshot(),7)
    data=decision.inference_diagnostics
    assert data['raw_green_sec']==raw
    assert data['bounded_green_sec']==bounded and data['rounded_green_sec']==rounded
    assert data['clipping_applied']==clipped and data['bound_touch_min']==touch
    assert data['rounding_applied']==rounding
    assert not data['backend_fallback'] and data['model_domain']=='unknown'
    assert engine.evaluate_green_time(0,0,0)==rounded  # legacy scalar API


def test_backend_failure_has_no_fabricated_raw_model_output():
    decision=ATSCDecisionEngine(synthetic_anfis(failure=True)).evaluate(TrafficSnapshot(),0)
    assert decision.green_seconds==30 and decision.failsafe_active
    data=decision.inference_diagnostics
    assert data['raw_green_sec'] is None and data['backend_fallback']
    assert data['backend_error'] and data['rounded_green_sec']==30


def test_decision_uses_returned_call_result_instead_of_last_error():
    engine=synthetic_anfis(10.004)
    original=engine.evaluate_green_time_result
    def poison(**kwargs):
        result=original(**kwargs)
        engine.last_error='different concurrent call failed'
        return result
    engine.evaluate_green_time_result=poison
    decision=ATSCDecisionEngine(engine).evaluate(TrafficSnapshot(),1)
    assert not decision.failsafe_active
    assert decision.inference_diagnostics['raw_green_sec']==10.004
