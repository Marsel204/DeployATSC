import pytest
from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.vehicle_waits import VehicleWaits


def vehicle(speed=0, point=(20,20), local_id=7, valid=True):
    return VehicleObservation(local_id,"car",speed,point,velocity_valid=valid)


def test_red_wait_and_total_stop_are_distinct_and_hysteresis_is_backdated():
    waits=VehicleWaits("run",RunPolicy(),side=lambda p:p[1]-100)
    for sequence in range(301):
        t=9+sequence/10
        speed=0 if 10<=t<34 else 2
        waits.observe([vehicle(speed)], "RED" if t<30 else "GREEN", t, sequence+1, 1)
    row=next(iter(waits.records.values()))
    assert row["red_wait_sec"] == pytest.approx(20,abs=1e-6)
    assert row["total_stopped_sec"] == pytest.approx(24,abs=1e-6)


def test_motion_unknown_and_duplicate_frames_cannot_inflate_wait():
    waits=VehicleWaits("run",RunPolicy(stop_persistence_sec=0),side=lambda p:p[1]-100)
    waits.observe([vehicle()],"RED",0,1,1)
    waits.observe([vehicle()],"RED",.1,2,1)
    for _ in range(5):waits.observe([vehicle()],"RED",.1,2,1)
    waits.observe([],"UNKNOWN",.2,3,1)
    waits.observe([vehicle()],"RED",.3,4,1)
    row=next(iter(waits.records.values()))
    assert row["red_wait_sec"] == pytest.approx(.1)
    assert row["unknown_sec"] >= .2-1e-8
    assert "observation_gap" in row["quality_flags"]


def test_wait_persists_over_cycles_and_crossing_counts_once():
    waits=VehicleWaits("run",RunPolicy(stop_persistence_sec=0),side=lambda p:p[1]-20)
    for seq in range(31):
        t=seq/10
        phase="RED" if t<1 or t>=2 else "GREEN"
        waits.observe([vehicle(point=(10,10))],phase,t,seq+1,1 if t<2 else 2)
    row=next(iter(waits.records.values()))
    assert row["red_wait_sec"] == pytest.approx(2)
    assert row["cycles"] == [1,2]
    waits.observe([vehicle(2,(10,25))],"GREEN",3.1,32,2)
    waits.observe([vehicle(2,(10,30))],"GREEN",3.2,33,2)
    assert row["crossed_stop_line"] and row["crossing_cycle"]==2
    assert len(waits.records)==1


def test_unstable_id_and_warmup_do_not_accrue_wait():
    waits=VehicleWaits("run",RunPolicy(stop_persistence_sec=0),side=lambda p:p[1]-100)
    invalid=VehicleObservation(7,"car",0,(10,10),stable_id=False,velocity_valid=True)
    waits.observe([invalid],"RED",0,1,1)
    assert not waits.records
    waits.observe([vehicle(valid=False)],"RED",.1,2,1)
    waits.observe([vehicle(valid=False)],"RED",.2,3,1)
    assert next(iter(waits.records.values()))["red_wait_sec"]==0


def test_crossing_at_waiting_area_exit_is_still_counted():
    waits=VehicleWaits("run",RunPolicy(stop_persistence_sec=0),side=lambda p:p[1]-20)
    waits.observe([vehicle(2,(10,15))],"GREEN",0,1,1)
    outside=VehicleObservation(7,"car",2,(10,25),velocity_valid=True,in_region=False)
    waits.observe([outside],"GREEN",.1,2,1)
    row=next(iter(waits.records.values()))
    assert row["crossed_stop_line"] and row["crossing_cycle"]==1


def test_export_snapshot_is_frozen_while_stop_confirmation_revises_live_intervals():
    from copy import deepcopy
    waits=VehicleWaits("run",RunPolicy(),side=lambda p:p[1]-100)
    waits.observe([vehicle()],"RED",0,1,1)
    waits.observe([vehicle()],"RED",.1,2,1)
    snapshot=waits.rows(include_intervals=True)
    frozen=deepcopy(snapshot)
    waits.observe([vehicle()],"RED",.2,3,1)
    waits.observe([vehicle()],"RED",.3,4,1)
    assert next(iter(waits.records.values()))["red_wait_sec"] == pytest.approx(.3)
    assert snapshot == frozen


def test_returning_track_with_native_velocity_history_remains_a_partial_journey(monkeypatch):
    from src.experiment.summary import summarize
    from src.perception.detector_tracker import DetectorTracker, TrackedVehicle

    monkeypatch.setattr(DetectorTracker, "_load_model", lambda self: None)
    tracker = DetectorTracker("unused.pt")
    waits = VehicleWaits("run", RunPolicy(), side=lambda p: p[1]-20)

    def observe(t, sequence, y):
        track = TrackedVehicle(7, 1, "car", [5, y-4, 15, y], 1, t)
        tracker.update_velocities([track], lambda x, y: (x, y), timestamp=t)
        waits.observe([VehicleObservation(7, "car", track.velocity_mps if track.velocity_valid else None,
                                         track.bottom_center, velocity_valid=track.velocity_valid)],
                      "GREEN", t, sequence, 1)

    for index in range(4):
        observe(index*.1, index+1, 10+index)
    for t, sequence in [(.8, 5), (1.3, 6), (1.8, 7)]:
        waits.observe([], "GREEN", t, sequence, 1)
    observe(2, 8, 15)
    observe(2.1, 9, 25)
    rows = waits.rows()
    assert len(rows) == 2
    assert "tracking_lost" in rows[0]["quality_flags"]
    assert rows[1]["crossed_stop_line"]
    assert rows[1]["left_censored"]
    summary = summarize(rows, [], None)
    assert summary["complete_vehicle_count"] == 0
    assert summary["partial_vehicle_count"] == 2
    assert summary["complete_mean_red_wait_sec"] is None


def test_new_moving_arrival_can_still_be_a_complete_journey():
    from src.experiment.summary import summarize

    waits = VehicleWaits("run", RunPolicy(), side=lambda p: p[1]-20)
    waits.observe([], "GREEN", 0, 1, 1)
    waits.observe([vehicle(2, (10, 10))], "GREEN", .1, 2, 1)
    waits.observe([vehicle(2, (10, 25))], "GREEN", .2, 3, 1)
    assert summarize(waits.rows(), [], None)["complete_vehicle_count"] == 1


@pytest.mark.parametrize('outside',[False,True])
def test_unknown_crossing_retains_evidence_without_guessed_cycle(outside):
    waits=VehicleWaits('run',RunPolicy(stop_persistence_sec=0),side=lambda p:p[1]-20)
    waits.observe([vehicle(2,(10,10))],'GREEN',0,1,1)
    crossing=VehicleObservation(7,'car',2,(10,25),velocity_valid=True,in_region=not outside)
    waits.observe([crossing],'UNKNOWN',.1,2,2) # even a legacy caller's guessed cycle is ignored
    waits.observe([vehicle(2,(10,30))],'GREEN',.2,3,2)
    row=waits.rows()[0]
    assert row['crossed_stop_line'] and row['crossing_cycle'] is None
    assert row['crossing_phase']=='UNKNOWN' and row['crossing_direction']==1
    assert row['completed_at']==.1 and row['crossing_at']==.1
    assert 2 not in row['cycles']
    assert 'crossing_phase_unknown' in row['quality_flags']


def test_unknown_phase_keeps_known_stopped_time_but_no_red_or_cycle_credit():
    waits=VehicleWaits('run',RunPolicy(stop_persistence_sec=0),side=lambda p:-1)
    for seq,(t,phase) in enumerate([(0,'UNKNOWN'),(.1,'UNKNOWN'),(.2,'GREEN')],1):
        waits.observe([vehicle()],phase,t,seq,2)
    row=waits.rows(include_intervals=True)[0]
    assert row['total_stopped_sec']==pytest.approx(.2)
    assert row['red_wait_sec']==0
    assert row['phase_unknown_sec']==pytest.approx(.2)
    assert row['unknown_sec']==0 # motion coverage remains independently known
    assert 'None' not in row['wait_by_cycle'] and not row['wait_by_cycle']
    assert row['unattributed_stopped_sec']==pytest.approx(.2)


def test_skipped_intervals_have_unknown_phase_coverage_without_cycle_credit():
    waits=VehicleWaits('run',RunPolicy(stop_persistence_sec=0),side=lambda p:-1)
    for seq,t in [(1,0),(2,.1),(4,.3),(5,1)]:
        waits.observe([vehicle()],'RED',t,seq,1)
    row=waits.rows(include_intervals=True)[0]
    assert row['red_wait_sec']==pytest.approx(.1)
    assert row['unknown_sec']==pytest.approx(.9)
    assert row['phase_unknown_sec']==pytest.approx(.9)
    assert all(i['phase']=='UNKNOWN' and i['cycle_id'] is None for i in row['intervals'] if not i['valid'])
