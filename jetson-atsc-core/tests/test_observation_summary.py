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


def test_summary_qualification_unknown_coverage_and_zero_denominators():
    rows=[{'red_wait_sec':0,'total_stopped_sec':.2,'unknown_sec':.3,'phase_unknown_sec':.1,
           'unattributed_stopped_sec':.2,'crossed_stop_line':True,'crossing_cycle':None,
           'completed':True,'left_censored':False,'right_censored':False,'quality_flags':['phase_unknown']}]
    result=summarize(rows,[{'eligible':True,'qualified':False},{'eligible':True,'qualified':True}],None)
    assert result['valid_cycle_comparisons']==2 and result['qualified_cycle_comparisons']==1
    assert result['qualified_comparison_fraction']==.5
    assert result['observed_wait_sample_count']==1 and result['complete_wait_sample_count']==0
    assert result['motion_unknown_sec']==.3 and result['phase_unknown_sec']==.1
    assert result['unattributed_crossing_count']==1 and result['unattributed_stopped_sec']==.2
    assert result['partial_vehicle_fraction']==1
    empty=summarize([],[],None)
    assert empty['qualified_comparison_fraction'] is None and empty['partial_vehicle_fraction'] is None
    assert empty['observed_mean_red_wait_sec'] is None
