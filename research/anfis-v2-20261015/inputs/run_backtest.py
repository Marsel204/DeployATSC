"""Held-out, controller-only point-queue counterfactual. No production code changes."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics
import time
import numpy as np
import onnxruntime as ort

HERE=Path(__file__).resolve().parent
CORE=HERE.parents[1]/'DeployATSC/jetson-atsc-core'
DATA=json.loads((HERE/'prepared_datasets.json').read_text())
CFG=json.loads((HERE/'input_manifest.json').read_text())

def write_csv(name,rows):
    assert rows,name
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def key(v):return v['direction']+':'+v['vehicle_id']

def jobs_for(ds,speed_mult=1.):
    out={d:[] for d in ['E','W']}
    for v in ds['vehicles']:
        v=dict(v)
        v['ready_sec']=v['boundary_arrival_sec']+(v['ready_sec']-v['boundary_arrival_sec'])/speed_mult
        v['recorded_approach_delay_sec']=max(0,v['recorded_entry_sec']-v['ready_sec'])
        out[v['direction']].append(v)
    # Simultaneous initial-stock arrivals use visible queue position, never future observed departures.
    return {d:sorted(vs,key=lambda v:(v['ready_sec'],v['first_observed_upstream_distance_m'],int(v['vehicle_id']))) for d,vs in out.items()}

def serve(vs,departures,cursor,previous,start,end,params):
    """One aggregate FIFO queue. An idle green permits an unqueued arrival immediately."""
    h,tau=params
    while cursor<len(vs):
        v=vs[cursor];ready=v['ready_sec']
        candidate=max(ready,previous+h*v['pcu'])
        if ready<start-1e-9:candidate=max(candidate,start+tau)
        else:candidate=max(candidate,start)
        if candidate>end+1e-9:break
        assert candidate>=ready-1e-8
        departures[key(v)]=candidate;previous=candidate;cursor+=1
    return cursor,previous

def fixed(ds,parameters,speed_mult=1.,yellow=3.):
    jobs=jobs_for(ds,speed_mult);departures={};horizon=ds['horizon']
    for d in ['E','W']:
        cursor=0;previous=-math.inf
        for g in ds['greens']:
            end=min(horizon,g['end']+yellow)
            if g['start']>horizon:break
            cursor,previous=serve(jobs[d],departures,cursor,previous,g['start'],end,parameters[d])
    return departures,jobs

def features(vs,departures,t,d,width_mult=1.,default_width=False):
    active=[v for v in vs if v['boundary_arrival_sec']<=t and key(v) not in departures]
    queued=[v for v in active if v['ready_sec']<=t]
    alpha=CFG['queue_tail_m_per_pcu'][d]
    # Observation-region capacity; excess queued vehicles remain in the virtual buffer.
    visible=[];pcu=0.;capacity=45/max(alpha,1e-6)
    for v in sorted(active,key=lambda q:q['ready_sec']):
        if pcu+v['pcu']>capacity+1e-9:break
        visible.append(v);pcu+=v['pcu']
    width=(10.5 if default_width else CFG['roi_width_m'][d])*width_mult
    return {'v_w_pcu':pcu,'queue_m':min(45,alpha*sum(v['pcu'] for v in queued)),
        'occupancy_pct':min(100,sum(v['footprint_m2'] for v in visible)/(45*width)*100),
        'visible_vehicles':len(visible),'queued_vehicles':len(queued),
        'pending_vehicles':len(active),'roi_width_m':width}

options=ort.SessionOptions();options.intra_op_num_threads=2
model=CORE/'models/anfis_sugeno.onnx'
session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
input_name=session.get_inputs()[0].name
def recommend(f):
    value=float(session.run(None,{input_name:np.array([[f['v_w_pcu'],f['queue_m'],f['occupancy_pct']]],np.float32)})[0][0,0])
    assert math.isfinite(value) and 10-1e-5<=value<=120+1e-5,value
    return round(float(np.clip(value,10,120)),2)

def adaptive(ds,parameters,speed_mult=1.,yellow=3.,width_mult=1.,default_width=False,identity=False):
    jobs=jobs_for(ds,speed_mult);departures={};cursors={d:0 for d in jobs};previous={d:-math.inf for d in jobs}
    greens=ds['greens'];t=greens[0]['start'];cycle=0;records=[];horizon=ds['horizon']
    red_intervals=[b['start']-a['end'] for a,b in zip(greens,greens[1:])]
    red_fallback=float(np.median(red_intervals))
    while t<horizon-1e-9:
        fs={d:features(jobs[d],departures,t,d,width_mult,default_width) for d in jobs}
        recs={d:recommend(fs[d]) for d in jobs}
        g=(greens[cycle]['end']-greens[cycle]['start']) if identity and cycle<len(greens) else max(recs.values())
        if identity and cycle>=len(greens):break
        service_end=min(horizon,t+g+yellow)
        for d in jobs:
            cursors[d],previous[d]=serve(jobs[d],departures,cursors[d],previous[d],t,service_end,parameters[d])
        red=red_intervals[cycle] if cycle<len(red_intervals) else red_fallback
        records.append({'date':ds['date'],'opportunity':cycle+1,'green_start_sec':t,'green_end_sec':min(horizon,t+g),
            'requested_green_sec':g,'red_after_green_sec':red,'red_interval_extrapolated':cycle>=len(red_intervals),
            'yellow_service_sec':yellow,
            **{d+'_'+k:v for d,f in fs.items() for k,v in f.items()},
            **{d+'_recommended_green_sec':v for d,v in recs.items()}})
        t+=g+red;cycle+=1
        assert cycle<1000
    return departures,jobs,records

def delay(v,deps,horizon):return max(0,min(horizon,deps.get(key(v),horizon))-v['ready_sec'])

def score(ds,baseline,anfis,jobs,scenario):
    out=[]
    for d,vs in jobs.items():
        for v in vs:
            if not v['eligible']:continue
            base=delay(v,baseline,ds['horizon']);counter=delay(v,anfis,ds['horizon']);ref=v['recorded_approach_delay_sec']
            out.append({'scenario':scenario,'date':ds['date'],'vehicle_id':v['vehicle_id'],'direction':d,
                'class':v['class'],'boundary_arrival_sec':v['boundary_arrival_sec'],'freeflow_ready_sec':v['ready_sec'],
                'recorded_entry_sec':v['recorded_entry_sec'],'recorded_approach_delay_sec':ref,
                'recorded_stopped_sec':v['recorded_stopped_sec'],
                'modeled_baseline_entry_sec':baseline.get(key(v)),
                'modeled_anfis_entry_sec':anfis.get(key(v)),
                'modeled_baseline_delay_sec':base,'modeled_anfis_delay_sec':counter,
                'direct_recorded_minus_anfis_sec':ref-counter,
                'paired_baseline_minus_anfis_sec':base-counter,
                'baseline_error_sec':base-ref,'baseline_served':key(v) in baseline,'anfis_served':key(v) in anfis,
                'horizon_sec':ds['horizon'],'right_censored_anfis':key(v) not in anfis})
    return out

def summarize(rows):
    if not rows:return {}
    mean=lambda k:float(np.mean([r[k] for r in rows]))
    mae=float(np.mean([abs(r['baseline_error_sec']) for r in rows]));within=float(np.mean([abs(r['baseline_error_sec'])<=10 for r in rows]))
    return {'vehicles':len(rows),'recorded_mean_approach_delay_sec':mean('recorded_approach_delay_sec'),
        'recorded_mean_stopped_sec':mean('recorded_stopped_sec'),
        'modeled_baseline_mean_delay_sec':mean('modeled_baseline_delay_sec'),
        'modeled_anfis_mean_delay_sec':mean('modeled_anfis_delay_sec'),
        'direct_recorded_minus_anfis_mean_sec':mean('direct_recorded_minus_anfis_sec'),
        'paired_baseline_minus_anfis_mean_sec':mean('paired_baseline_minus_anfis_sec'),
        'paired_total_savings_vehicle_sec':sum(r['paired_baseline_minus_anfis_sec'] for r in rows),
        'paired_savings_pct':100*sum(r['paired_baseline_minus_anfis_sec'] for r in rows)/sum(r['modeled_baseline_delay_sec'] for r in rows),
        'baseline_mae_sec':mae,'baseline_bias_sec':mean('baseline_error_sec'),
        'baseline_p90_abs_error_sec':float(np.percentile([abs(r['baseline_error_sec']) for r in rows],90)),
        'baseline_fraction_within_10_sec':within,
        'baseline_gate_pass':bool(mae<=5 and within>=.8),
        'baseline_unserved':sum(not r['baseline_served'] for r in rows),
        'anfis_unserved':sum(not r['anfis_served'] for r in rows),
        'improved_vehicles':sum(r['paired_baseline_minus_anfis_sec']>.1 for r in rows),
        'worsened_vehicles':sum(r['paired_baseline_minus_anfis_sec']<-.1 for r in rows)}

parameters={};surface=[];train=DATA[0]
for d in ['E','W']:
    best=None
    for h in np.arange(.3,2.51,.1):
        for startup in np.arange(0,5.01,.5):
            params={x:(float(h),float(startup)) for x in ['E','W']}
            deps,jobs=fixed(train,params)
            vals=[abs(delay(v,deps,train['horizon'])-v['recorded_approach_delay_sec']) for v in jobs[d] if v['eligible']]
            loss=float(np.mean(vals));r={'direction':d,'headway_sec_per_pcu':float(h),'startup_sec':float(startup),'calibration_mae_sec':loss}
            surface.append(r)
            if best is None or loss<best['calibration_mae_sec']-1e-9:best=r
    parameters[d]=(best['headway_sec_per_pcu'],best['startup_sec'])
    print('FITTED',json.dumps(best),flush=True)
write_csv('calibration_loss_surface.csv',surface)
(HERE/'fitted_parameters.json').write_text(json.dumps({'parameters':parameters,'calibration_date':train['date'],
    'training_only_geometry':{k:CFG[k] for k in ['freeflow_mps','queue_tail_m_per_pcu','roi_width_m']},
    'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'onnxruntime':ort.__version__},indent=2))

checks=[]
for name,ready,start,end,startup,expected in [
    ('unqueued_green_arrival_has_zero_queue_delay',5.,0.,10.,2.,5.),
    ('red_arrival_waits_for_green_and_startup',3.,10.,20.,2.,12.)]:
    v={'direction':'E','vehicle_id':'sanity','ready_sec':ready,'pcu':1.}
    dep={};serve([v],dep,0,-math.inf,start,end,(1.,startup))
    assert abs(dep[key(v)]-expected)<1e-9,name
    checks.append({'date':'synthetic_logic_check','check':name,'max_difference_sec':abs(dep[key(v)]-expected),'passed':True})
for ds in DATA:
    fixed_deps,jobs=fixed(ds,parameters)
    identity_deps,_,_=adaptive(ds,parameters,identity=True)
    assert set(fixed_deps)==set(identity_deps),(ds['date'],'identity membership')
    err=max([abs(v-identity_deps[k]) for k,v in fixed_deps.items()] or [0])
    assert err<1e-6,(ds['date'],err)
    checks.append({'date':ds['date'],'check':'unchanged_green_policy_reproduces_baseline','max_difference_sec':err,'passed':True})
write_csv('verification_checks.csv',checks)

scenarios=[('primary',1.,0.,1.,3.,1.,False),
    ('headway_minus20pct',.8,0.,1.,3.,1.,False),('headway_plus20pct',1.2,0.,1.,3.,1.,False),
    ('startup_minus1sec',1.,-1.,1.,3.,1.,False),('startup_plus1sec',1.,1.,1.,3.,1.,False),
    ('freeflow_minus20pct',1.,0.,.8,3.,1.,False),('freeflow_plus20pct',1.,0.,1.2,3.,1.,False),
    ('no_yellow_service',1.,0.,1.,0.,1.,False),
    ('roi_width_minus25pct',1.,0.,1.,3.,.75,False),('roi_width_plus25pct',1.,0.,1.,3.,1.25,False),
    ('app_default_roi_width',1.,0.,1.,3.,1.,True)]
allrows=[];cycles=[];summaries=[]
for scenario,hm,ts,vm,yellow,wm,default_width in scenarios:
    params={d:(h*hm,max(0,t+ts)) for d,(h,t) in parameters.items()}
    for ds in DATA:
        base,jobs=fixed(ds,params,vm,yellow)
        new,_,logs=adaptive(ds,params,vm,yellow,wm,default_width)
        rows=score(ds,base,new,jobs,scenario);allrows.extend(rows)
        cycles.extend([{'scenario':scenario,**r} for r in logs])
        for direction in ['ALL','E','W']:
            selected=rows if direction=='ALL' else [r for r in rows if r['direction']==direction]
            summaries.append({'scenario':scenario,'date':ds['date'],'split':'calibration' if ds['date']==train['date'] else 'heldout',
                'direction':direction,**summarize(selected)})
        if scenario=='primary':print('RESULT',ds['date'],json.dumps(summarize(rows)),flush=True)
    held=[r for r in allrows if r['scenario']==scenario and r['date'] in CFG['heldout_dates']]
    summaries.append({'scenario':scenario,'date':'combined_heldout','split':'heldout','direction':'ALL',**summarize(held)})
write_csv('vehicle_delay_comparisons.csv',allrows)
write_csv('primary_vehicle_delays.csv',[r for r in allrows if r['scenario']=='primary'])
write_csv('cycle_recommendations.csv',cycles)
write_csv('scenario_summary.csv',summaries)
(HERE/'results.json').write_text(json.dumps({'summaries':summaries,'parameters':parameters,'checks':checks,
    'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
print('BACKTEST_COMPLETE',len(allrows),'vehicle comparisons',len(cycles),'phase opportunities',flush=True)
