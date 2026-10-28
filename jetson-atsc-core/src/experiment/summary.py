from __future__ import annotations
import statistics


def _percentile(values, p):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*p
    lo=int(index);hi=min(lo+1,len(values)-1)
    return round(values[lo]+(values[hi]-values[lo])*(index-lo),6)


def summarize(rows,comparisons,method,predictions=()):
    rows=list(rows)
    comparisons=list(comparisons)
    predictions=list(predictions)
    complete=[r for r in rows if r.get("completed") and not r.get("left_censored",True)
              and not r.get("right_censored",True) and not r.get("quality_flags")]
    values=[r.get("red_wait_sec",0.) for r in rows]
    finished=[r.get("red_wait_sec",0.) for r in complete]
    timing_count=sum(bool(c.get("eligible")) for c in comparisons)
    qualified_count=sum(bool(c.get("qualified")) for c in comparisons)
    return {"operation":"observation", "observed_vehicle_count":len(rows),
        "complete_vehicle_count":len(complete),"partial_vehicle_count":len(rows)-len(complete),
        "observed_mean_red_wait_sec":round(statistics.mean(values),6) if values else None,
        "observed_median_red_wait_sec":_percentile(values,.5),
        "observed_p95_red_wait_sec":_percentile(values,.95),
        "complete_mean_red_wait_sec":round(statistics.mean(finished),6) if finished else None,
        "valid_cycle_comparisons":timing_count,
        "qualified_cycle_comparisons":qualified_count,
        "observed_cycle_count":len(comparisons),
        "incomplete_cycle_count":sum(not bool(c.get("complete")) for c in comparisons),
        "qualified_comparison_fraction":qualified_count/timing_count if timing_count else None,
        "timing_eligible_prediction_count":sum(bool(p.get("eligible")) for p in predictions),
        "qualified_prediction_count":sum(bool(p.get("qualified")) for p in predictions),
        "unknown_qualification_prediction_count":sum(not p.get("qualification") or p["qualification"].get("state")=="unknown" for p in predictions),
        "observed_wait_sample_count":len(values),"complete_wait_sample_count":len(finished),
        "partial_vehicle_fraction":(len(rows)-len(complete))/len(rows) if rows else None,
        "motion_unknown_sec":round(sum(r.get("unknown_sec",0.) for r in rows),6),
        "phase_unknown_sec":round(sum(r.get("phase_unknown_sec",0.) for r in rows),6),
        "motion_unknown_vehicle_count":sum(r.get("unknown_sec",0)>0 for r in rows),
        "phase_unknown_vehicle_count":sum(r.get("phase_unknown_sec",0)>0 for r in rows),
        "unattributed_crossing_count":sum(bool(r.get("crossed_stop_line") and r.get("crossing_cycle") is None) for r in rows),
        "unattributed_stopped_sec":round(sum(r.get("unattributed_stopped_sec",0.) for r in rows),6),
        "coverage_definition":"Durations sum vehicle-seconds across observed track intervals; motion and phase unknown may overlap. Qualified fraction uses timing-eligible comparisons; partial fraction uses all observed tracks. Zero denominators are null.",
        "sample_definition":"All observed acquisition tracks; incomplete waits are lower bounds. Complete journeys reported separately.",
        "los":{"method":method,"grade":None,
            "reason":"LOS ungraded: a selected control-delay methodology, edition and sufficient measurement coverage are required"}}
