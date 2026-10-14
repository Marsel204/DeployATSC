import io,json,zipfile
from src.experiment.export import ExperimentStore


def test_versioned_export_preserves_linked_evidence(tmp_path):
    store=ExperimentStore(tmp_path,"run",{"operation":"observation","sources":{"system":"road.mp4","acquisition":"signal.mp4"}})
    state={"run_id":"run","predictions":[{"request_id":"r","cycle_id":1}],
           "phase_events":[{"phase":"GREEN","timestamp":30}],
           "vehicles":[{"vehicle_id":"run:acquisition:0:7","red_wait_sec":20,"intervals":[{"start":10,"end":30}]}],
           "comparisons":[{"cycle_id":1,"actual_green_sec":25}],"closed":True}
    archive=store.export(state)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest=json.loads(bundle.read("manifest.json"))
        assert manifest["operation"]=="observation"
        assert "vehicles.csv" in bundle.namelist() and "evidence.json" in bundle.namelist()
        data=json.loads(bundle.read("evidence.json"))
        assert data["vehicles"][0]["intervals"][0]["end"]==30
        assert data["predictions"][0]["cycle_id"]==data["comparisons"][0]["cycle_id"]
