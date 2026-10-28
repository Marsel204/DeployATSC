import threading
import pytest
from src.common.types import TrafficSnapshot, ActuationDecision
from src.experiment.contracts import RunPolicy
from src.experiment.predictions import PredictionBook, PredictionConflict


def test_duplicate_capture_and_pending_conflict():
    calls=[]
    def evaluate(snapshot,cycle):
        calls.append(snapshot)
        return ActuationDecision(cycle,40)
    book=PredictionBook("run",RunPolicy(),evaluate,clock=lambda:0)
    snapshot=TrafficSnapshot(timestamp=0,v_w_pcu=7)
    book.capture(snapshot,0,"RED","click-1")
    book.await_idle()
    duplicate=book.capture(snapshot,0,"RED","click-1")
    assert len(calls)==1 and duplicate["snapshot"]["v_w_pcu"]==7
    with pytest.raises(PredictionConflict):book.capture(snapshot,0,"RED","click-2")
    book.close()


def test_green_during_inference_is_late_not_a_valid_recommendation():
    release=threading.Event();started=threading.Event();now={"t":0}
    def evaluate(snapshot,cycle):
        started.set();release.wait(2)
        return ActuationDecision(cycle,40)
    book=PredictionBook("run",RunPolicy(),evaluate,clock=lambda:now["t"])
    book.capture(TrafficSnapshot(timestamp=0),0,"RED","click")
    assert started.wait(1)
    now["t"]=1;book.on_green(1,1)
    now["t"]=2;release.set();book.await_idle()
    record=book.rows()[0]
    assert record["status"]=="late" and not record["eligible"]
    assert record["completed_lead_sec"]==-1
    book.close()


def test_stale_green_and_cancelled_captures():
    book=PredictionBook("run",RunPolicy(),lambda s,c:ActuationDecision(c,30),clock=lambda:0)
    with pytest.raises(ValueError):book.capture(TrafficSnapshot(timestamp=0),2,"RED","stale")
    with pytest.raises(PredictionConflict):book.capture(TrafficSnapshot(timestamp=0),0,"GREEN","green")
    book.capture(TrafficSnapshot(timestamp=0),0,"RED","cancel")
    book.cancel("cancel");book.await_idle();book.on_green(.2,1)
    assert book.rows()[0]["status"]=="cancelled"
    assert book.rows()[0]["cycle_id"] is None
    book.close()


def test_anfis_backend_fallback_cannot_be_an_eligible_recommendation():
    from src.control.decision_engine import ATSCDecisionEngine
    class Fallback:
        last_error="ONNX backend unavailable"
        def evaluate_green_time(self,**kwargs):return 30
    decision=ATSCDecisionEngine(Fallback())
    book=PredictionBook("run",RunPolicy(),decision.evaluate,clock=lambda:0)
    book.capture(TrafficSnapshot(timestamp=0),0,"RED","failure")
    book.await_idle()
    assert book.rows()[0]["status"]=="failed"
    assert not book.rows()[0]["eligible"]
    book.close()


@pytest.mark.parametrize('raw,failure',[(-5.,False),(10.,False),(10.004,False),(30.,True)])
def test_capture_retains_per_call_diagnostics_even_on_backend_fallback(raw,failure):
    from test_decision_engine import synthetic_anfis
    from src.control.decision_engine import ATSCDecisionEngine
    decision=ATSCDecisionEngine(synthetic_anfis(raw,failure))
    book=PredictionBook('diagnostics',RunPolicy(),decision.evaluate,clock=lambda:0)
    try:
        book.capture(TrafficSnapshot(timestamp=0),0,'RED','capture')
        book.await_idle();book.on_green(1,1)
        row=book.rows()[0]
        data=row['inference_diagnostics']
        assert data['raw_green_sec']==(None if failure else raw)
        assert data['backend_fallback']==failure
        assert data['model_domain']=='unknown'
        assert row['eligible']==(not failure)
        assert row['status']==('failed' if failure else 'matched')
        if raw==-5:assert data['clipping_applied']
        if raw==10:assert data['bound_touch_min'] and not data['clipping_applied']
        if raw==10.004:assert data['rounding_applied'] and data['rounded_to_bound'] and not data['bound_touch_min']
    finally:book.close()
