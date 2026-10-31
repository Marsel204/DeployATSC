from test_persistent_identity_pipeline import engine,tick
from test_vehicle_identity import detection


def test_admission_retains_first_detection_and_native_provenance():
    book=engine()
    try:
        for seq in range(1,11):tick(book,seq,(seq-1)/10,(detection(4 if seq==1 else 228),))
        row=book.snapshot(evidence=True)['vehicles'][0]
        assert row.get('first_detected_at')==0.
        assert row.get('admitted_at')==.1
        assert row.get('first_native_reference')=={'epoch':0,'local_id':4}
    finally:book.close()


def test_ambiguous_gap_is_visible_and_never_receives_hidden_wait_credit():
    book=engine()
    try:
        for seq in range(1,11):tick(book,seq,(seq-1)/10,(detection(4,40),detection(5,60)))
        tick(book,11,1.,(detection(99,50),))
        rows=book.snapshot()['vehicles']
        assert len(rows)==2
        assert all(r.get('identity_state')=='AMBIGUOUS' for r in rows)
        assert all(r['pending_estimated_red_wait_sec']==0 for r in rows)
        for seq in range(12,21):tick(book,seq,(seq-1)/10,(detection(4,40),detection(5,60)))
        assert all(r['estimated_red_wait_sec']==0 for r in book.snapshot()['vehicles'])
    finally:book.close()


def test_unverified_metric_projection_cannot_veto_supported_image_matching():
    from dataclasses import replace
    from src.experiment.contracts import RunPolicy
    from src.experiment.engine import ObservationEngine
    from src.experiment.identity_contracts import IdentityPolicy
    book=ObservationEngine('unscaled',RunPolicy(debounce_frames=1),
        {'xmin':0,'ymin':0,'xmax':10,'ymax':10},lambda p:-1,lambda *_:None,
        scale_evidence={'acquisition':{'state':'review_required'}},
        identity_policy=IdentityPolicy(mode='persistent_v1',confirmation_observations=2,confirmation_sec=.1))
    try:
        for seq in range(1,11):result=tick(book,seq,(seq-1)/10,(replace(detection(),ground=(0,0)),))
        identity=result.vehicles[0].logical_id
        for seq in range(11,15):result=tick(book,seq,(seq-1)/10,(replace(detection(228,x=51),ground=(4,0)),))
        assert result.vehicles[0].logical_id==identity
        assert book.snapshot()['vehicle_identity']['association_space']=='image'
    finally:book.close()
