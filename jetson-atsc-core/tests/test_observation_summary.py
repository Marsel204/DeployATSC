import pytest
from src.experiment.summary import summarize


def test_wait_statistics_include_zero_wait_and_disclose_partial_records():
    rows=[{"red_wait_sec":v,"total_stopped_sec":v+2,"completed":True,
           "left_censored":False,"right_censored":False,"quality_flags":[]} for v in [0,10,20,30]]
    result=summarize(rows,[],None)
    assert result["observed_mean_red_wait_sec"]==15
    assert result["observed_median_red_wait_sec"]==15
    assert result["observed_p95_red_wait_sec"]==pytest.approx(28.5)
    assert result["complete_vehicle_count"]==4
    rows.append({"red_wait_sec":100,"total_stopped_sec":100,"completed":True,"left_censored":True,"right_censored":True,"quality_flags":["tracking_lost"]})
    result=summarize(rows,[],None)
    assert result["partial_vehicle_count"]==1 and result["complete_mean_red_wait_sec"]==15
    assert result["observed_vehicle_count"]==5


def test_red_wait_does_not_generate_los_or_causal_improvement():
    result=summarize([{"red_wait_sec":1,"completed":True}],[],"HCM")
    assert result["los"]["grade"] is None
    assert "control" in result["los"]["reason"].lower()
    assert result["operation"]=="observation"
