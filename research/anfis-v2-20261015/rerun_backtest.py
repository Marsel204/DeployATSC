"""Frozen vehicle-by-vehicle FLUID point-queue adapter, old versus new model.

Reuses reviewed historical function definitions without executing their mutating
top-level driver. No calibration, controller or geometry tuning is performed.
"""
import ast
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
HERE = None  # A new output directory, chosen explicitly by the operator.
SOURCE = ROOT/'inputs'
NEW = ROOT/'model/candidate_anfis.onnx'
OLD = SOURCE/'old_deployed_anfis.onnx'
LOS_SOURCE = 'https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(name, rows):
    assert rows, name
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def delay_band(seconds):
    # Only an indicative band for the restricted-corridor delay proxy. LOS is
    # defined on movement/approach aggregate control delay, not each vehicle.
    for upper, band in [(10,'A'),(20,'B'),(35,'C'),(55,'D'),(80,'E')]:
        if seconds <= upper: return band
    return 'F'


def main():
    global HERE
    if not __debug__:
        raise RuntimeError('Run without -O: scientific consistency checks must remain enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory; existing directories are never overwritten')
    parser.add_argument('--raw-data-root', type=Path,
                        help='Optional original FLUID derived_data root for nine raw CSV hash checks')
    args = parser.parse_args()
    from verify_bundle import verify_bundle
    verify_bundle()
    HERE = args.output.expanduser().resolve()
    if HERE == ROOT or (ROOT in HERE.parents and not (ROOT/'generated') in HERE.parents):
        parser.error('Inside the bundle, output must be below generated/')
    HERE.mkdir(parents=True, exist_ok=False)
    raw_sources_verified = False
    started=time.monotonic()
    assert not (HERE/'results.json').exists(), 'Use a fresh directory; preserve completed results'
    data=json.loads((SOURCE/'prepared_datasets.json').read_text())
    config=json.loads((SOURCE/'input_manifest.json').read_text())
    fitted=json.loads((SOURCE/'fitted_parameters.json').read_text())
    parameters={k:tuple(v) for k,v in fitted['parameters'].items()}
    source_files=[p for p in SOURCE.iterdir() if p.is_file()]
    original_hashes={str(p):sha(p) for p in source_files}
    assert sha(OLD)==fitted['model_sha256'], 'Historical ONNX identity changed'
    if args.raw_data_root is not None:
        for relative, expected in config['sha256'].items():
            assert sha(args.raw_data_root/relative)==expected, relative
        raw_sources_verified = True
    export=json.loads((NEW.parent/'export_manifest.json').read_text())
    assert sha(NEW)==export['onnx_sha256'] and export['development_only']
    tree=ast.parse((SOURCE/'run_backtest.py').read_text())
    names={'key','jobs_for','serve','fixed','features','adaptive','delay','score','summarize'}
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {n.name for n in definitions}==names
    ns=dict(np=np, math=math, statistics=statistics, CFG=config)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(SOURCE/'run_backtest.py'),'exec'),ns)
    scenarios=ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='scenarios' for target in node.targets)))
    sessions={}
    options=ort.SessionOptions(); options.intra_op_num_threads=2
    for label, model in [('old',OLD),('new',NEW)]:
        session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        sessions[label]=session
    active=['old']
    def recommend(features):
        session=sessions[active[0]]
        x=np.array([[features[k] for k in ('v_w_pcu','queue_m','occupancy_pct')]],np.float32)
        value=float(session.run(None,{session.get_inputs()[0].name:x})[0][0,0])
        assert math.isfinite(value) and 10-1e-5 <= value <= 120+1e-5
        return round(float(np.clip(value,10,120)),2)
    ns['recommend']=recommend
    checks=[]
    for name, ready, start, end, startup, expected in [
        ('unqueued_green_arrival_has_zero_queue_delay',5.,0.,10.,2.,5.),
        ('red_arrival_waits_for_green_and_startup',3.,10.,20.,2.,12.)]:
        dep={}; v=dict(direction='E',vehicle_id='sanity',ready_sec=ready,pcu=1.)
        ns['serve']([v],dep,0,-math.inf,start,end,(1.,startup))
        assert abs(dep[ns['key'](v)]-expected)<1e-9
        checks.append(dict(check=name,passed=True))
    for ds in data:
        baseline,jobs=ns['fixed'](ds,parameters)
        for controller in sessions:
            active[0]=controller
            identity,_,_=ns['adaptive'](ds,parameters,identity=True)
            assert set(identity)==set(baseline)
            assert max((abs(identity[k]-v) for k,v in baseline.items()),default=0)<1e-6
            checks.append(dict(date=ds['date'],controller=controller,check='identity_schedule',passed=True))

    with (SOURCE/'vehicle_delay_comparisons.csv').open() as f:
        historical={(r['scenario'],r['date'],r['direction'],r['vehicle_id']):r for r in csv.DictReader(f)}
    comparisons=[]; cycles=[]; model_audit=[]; legacy_max_error=0.; checked=0
    for scenario,hm,ts,vm,yellow,wm,default_width in scenarios:
        params={d:(h*hm,max(0,t+ts)) for d,(h,t) in parameters.items()}
        for ds in data:
            base,jobs=ns['fixed'](ds,params,vm,yellow)
            outputs={}; scores={}
            for controller in sessions:
                active[0]=controller
                departures,_,logs=ns['adaptive'](ds,params,vm,yellow,wm,default_width)
                outputs[controller]=departures
                scored=ns['score'](ds,base,departures,jobs,scenario)
                scores[controller]={(r['direction'],r['vehicle_id']):r for r in scored}
                cycles.extend(dict(controller=controller,scenario=scenario,**r) for r in logs)
            assert set(scores['old'])==set(scores['new'])
            for identity, old in scores['old'].items():
                new=scores['new'][identity]
                previous=historical[(scenario,ds['date'],*identity)]
                for name in ('modeled_baseline_delay_sec','modeled_anfis_delay_sec','recorded_approach_delay_sec'):
                    difference=abs(old[name]-float(previous[name]))
                    legacy_max_error=max(legacy_max_error,difference)
                    assert difference<1e-6, (scenario,ds['date'],identity,name,difference)
                assert str(old['anfis_served'])==previous['anfis_served']
                checked+=1
                row={k:new[k] for k in ('scenario','date','vehicle_id','direction','class','boundary_arrival_sec',
                    'freeflow_ready_sec','recorded_entry_sec','recorded_approach_delay_sec','recorded_stopped_sec',
                    'modeled_baseline_entry_sec','modeled_baseline_delay_sec','baseline_error_sec','baseline_served','horizon_sec')}
                row.update(split='calibration' if ds['date']==config['calibration_date'] else 'reused_development_test',
                    old_model_entry_sec=old['modeled_anfis_entry_sec'],new_model_entry_sec=new['modeled_anfis_entry_sec'],
                    old_model_delay_sec=old['modeled_anfis_delay_sec'],new_model_delay_sec=new['modeled_anfis_delay_sec'],
                    recorded_minus_new_sec=new['direct_recorded_minus_anfis_sec'],
                    modeled_baseline_minus_new_sec=new['paired_baseline_minus_anfis_sec'],
                    old_model_minus_new_sec=old['modeled_anfis_delay_sec']-new['modeled_anfis_delay_sec'],
                    old_model_served=old['anfis_served'],new_model_served=new['anfis_served'],
                    right_censored_old=old['right_censored_anfis'],right_censored_new=new['right_censored_anfis'])
                comparisons.append(row)
            if scenario=='primary':
                for direction, vehicles in jobs.items():
                    for v in vehicles:
                        identity=ns['key'](v)
                        model_audit.append(dict(date=ds['date'],direction=direction,vehicle_id=v['vehicle_id'],
                            eligible_for_scoring=v['eligible'],exclusion_reason=v['exclusion_reason'],
                            boundary_arrival_sec=v['boundary_arrival_sec'],freeflow_ready_sec=v['ready_sec'],
                            recorded_entry_sec=v['recorded_entry_sec'],
                            baseline_entry_sec=base.get(identity),old_model_entry_sec=outputs['old'].get(identity),
                            new_model_entry_sec=outputs['new'].get(identity),
                            baseline_served=identity in base,old_model_served=identity in outputs['old'],
                            new_model_served=identity in outputs['new']))
    assert checked==len(historical)
    checks.append(dict(check='historical_all_vehicle_results_reproduced',rows=checked,max_error_sec=legacy_max_error,passed=True))

    def summary(rows, scenario, date, direction):
        mean=lambda name:float(np.mean([r[name] for r in rows]))
        base=mean('modeled_baseline_delay_sec'); old=mean('old_model_delay_sec'); new=mean('new_model_delay_sec')
        error=[abs(r['baseline_error_sec']) for r in rows]
        gate=float(np.mean(error))<=5 and float(np.mean(np.array(error)<=10))>=.8
        return dict(scenario=scenario,date=date,direction=direction,vehicles=len(rows),
            recorded_mean_delay_sec=mean('recorded_approach_delay_sec'),
            baseline_mean_delay_sec=base,old_mean_delay_sec=old,new_mean_delay_sec=new,
            recorded_minus_new_sec=mean('recorded_minus_new_sec'),
            baseline_minus_new_sec=base-new,old_minus_new_sec=old-new,
            delay_reduction_vs_baseline_percent=100*(base-new)/base,
            delay_reduction_vs_old_percent=100*(old-new)/old,
            baseline_mae_sec=float(np.mean(error)),baseline_within_10_sec=float(np.mean(np.array(error)<=10)),
            baseline_screen_pass=gate,
            baseline_unfinished=sum(not r['baseline_served'] for r in rows),
            old_unfinished=sum(not r['old_model_served'] for r in rows),new_unfinished=sum(not r['new_model_served'] for r in rows),
            improved_vs_baseline=sum(r['modeled_baseline_minus_new_sec']>.1 for r in rows),
            worsened_vs_baseline=sum(r['modeled_baseline_minus_new_sec']<-.1 for r in rows),
            improved_vs_old=sum(r['old_model_minus_new_sec']>.1 for r in rows),
            worsened_vs_old=sum(r['old_model_minus_new_sec']<-.1 for r in rows),
            baseline_delay_band=delay_band(base),old_delay_band=delay_band(old),new_delay_band=delay_band(new),
            verified_los=False)
    summaries=[]
    for scenario,*_ in scenarios:
        for date in [d['date'] for d in data]+['combined_reused_test']:
            selected=[r for r in comparisons if r['scenario']==scenario and
                (r['date']==date if date!='combined_reused_test' else r['date'] in config['heldout_dates'])]
            for direction in ['ALL','E','W']:
                rows=selected if direction=='ALL' else [r for r in selected if r['direction']==direction]
                if rows:summaries.append(summary(rows,scenario,date,direction))
    primary=[r for r in comparisons if r['scenario']=='primary']
    held=[r for r in primary if r['date'] in config['heldout_dates']]
    assert len(held)==136 and len(model_audit)==333
    assert len({(r['date'],r['direction'],r['vehicle_id']) for r in held})==136
    write_csv('heldout_vehicle_delays.csv',held)
    write_csv('primary_vehicle_delays.csv',primary)
    write_csv('all_scenario_vehicle_delays.csv',comparisons)
    write_csv('all_modeled_vehicles_audit.csv',model_audit)
    write_csv('cycle_recommendations.csv',cycles)
    write_csv('scenario_summary.csv',summaries)
    headline=[r for r in summaries if r['scenario']=='primary' and r['direction']=='ALL']
    write_csv('headline_results.csv',headline)
    write_json('verification_checks.json',checks)
    assert all(sha(path)==digest for path,digest in original_hashes.items())
    assert sha(OLD)==fitted['model_sha256'] and sha(NEW)==export['onnx_sha256']
    manifest=dict(original_experiment=str(SOURCE),original_files_sha256=original_hashes,
        raw_sources_verified=raw_sources_verified,
        rerun_basis='Bundled prepared vehicle data; raw CSVs checked only with --raw-data-root',
        source_data_sha256=config['sha256'],old_model_path=str(OLD),old_model_sha256=sha(OLD),
        new_model_path=str(NEW),new_model_sha256=sha(NEW),new_checkpoint_sha256=export['checkpoint_sha256'],
        calibration_reused_without_refitting=fitted,scenarios=scenarios,
        adapter_definitions_source_sha256=sha(SOURCE/'run_backtest.py'),runner_sha256=sha(__file__),
        los_band_source=LOS_SOURCE,los_band_basis='FHWA 2008 Traffic Signal Timing Manual Table 3-3 (HCM 2000)',
        los_note='Indicative delay bands only; 45 m restricted-corridor proxy, selected E/W cohort, incomplete movement/acceleration/capacity coverage. No verified whole-intersection LOS.',
        data_note='Previously inspected FLUID recordings reused for development comparison, not fresh confirmation.',
        onnxruntime=ort.__version__,elapsed_seconds=time.monotonic()-started)
    write_json('manifest.json',manifest)
    write_json('results.json',dict(status='complete',headline=headline,summaries=summaries,checks=checks,
        exclusions={d['date']:d['exclusions'] for d in data},manifest=manifest))
    # Reproduce every recorded controller output, not just a pooled headline.
    for name in ('heldout_vehicle_delays.csv', 'primary_vehicle_delays.csv',
                 'all_scenario_vehicle_delays.csv', 'all_modeled_vehicles_audit.csv',
                 'cycle_recommendations.csv', 'scenario_summary.csv', 'headline_results.csv'):
        assert (HERE/name).read_bytes() == (ROOT/'evidence/backtest'/name).read_bytes(), name
    print(json.dumps(dict(status='complete',elapsed_seconds=manifest['elapsed_seconds'],headline=headline),indent=2))


if __name__=='__main__':main()
