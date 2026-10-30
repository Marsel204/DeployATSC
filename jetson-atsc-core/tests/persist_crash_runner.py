"""Subprocess only: killed by its test while a real archive commit is in flight."""
import json,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_persist_background import Manual
from test_observation_sessions import FakeDetector,FakeANFIS


class Gated(FakeDetector):
    def detect_once(self,*a,**kw):threading.Event().wait(60);return []


payload,root,ready=sys.argv[1:];payload=json.loads(Path(payload).read_text())
s=Manual(payload,root,detector_factory=Gated,anfis_factory=FakeANFIS)
s.steps=1;s._process_batch();s.predict('pending-at-crash')
for _ in range(10):s.steps=1;s._process_batch()
original=s.store._stage

def stage(name,operation):
    if name=='archive_fsync':
        result=original(name,operation)
        Path(ready).write_text(json.dumps({'run_id':s.run_id,'processed':s.store.read()['processed_frames']}))
        threading.Event().wait(60)
        return result
    return original(name,operation)

s.store._stage=stage;s.applied_state='running';s._request_persist()
threading.Event().wait(60)
