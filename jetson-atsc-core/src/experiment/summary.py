from __future__ import annotations
import statistics


def _percentile(values, p):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*p
    lo=int(index);hi=min(lo+1,len(values)-1)
    return round(values[lo]+(values[hi]-values[lo])*(index-lo),6)


def summarize(rows,comparisons,method):
    rows=list(rows)
    complete=[r for r in rows if r.get("completed") and not r.get("left_censored",True)
              and not r.get("right_censored",True) and not r.get("quality_flags")]
    values=[r.get("red_wait_sec",0.) for r in rows]
    finished=[r.get("red_wait_sec",0.) for r in complete]
    return {"operation":"observation", "observed_vehicle_count":len(rows),
        "complete_vehicle_count":len(complete),"partial_vehicle_count":len(rows)-len(complete),
        "observed_mean_red_wait_sec":round(statistics.mean(values),6) if values else None,
        "observed_median_red_wait_sec":_percentile(values,.5),
        "observed_p95_red_wait_sec":_percentile(values,.95),
        "complete_mean_red_wait_sec":round(statistics.mean(finished),6) if finished else None,
        "valid_cycle_comparisons":sum(bool(c["eligible"]) for c in comparisons),
        "sample_definition":"All observed acquisition tracks; incomplete waits are lower bounds. Complete journeys reported separately.",
        "los":{"method":method,"grade":None,
            "reason":"LOS ungraded: a selected control-delay methodology, edition and sufficient measurement coverage are required"}}
