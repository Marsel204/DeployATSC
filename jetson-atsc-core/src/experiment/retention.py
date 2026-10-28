"""Acknowledged eviction and additive aggregates for the live observation path.

Historical reconstruction is deliberately an explicit reader operation. Live
quantiles become unavailable once samples leave memory; the reader computes
exact quantiles for a pinned generation on request.
"""
from collections import deque
from copy import deepcopy
from src.experiment.archive_index import ArchiveIndex
from src.experiment.summary import summarize

DURATIONS={'motion_unknown_sec':'unknown_sec','phase_unknown_sec':'phase_unknown_sec','unattributed_stopped_sec':'unattributed_stopped_sec'}


def aggregate(rows,comparisons=(),predictions=()):
    rows=list(rows);result=summarize(rows,comparisons,None,predictions)
    for key,field in DURATIONS.items():result[key]=sum(r.get(field,0.) for r in rows)
    return result


COUNTS=('observed_vehicle_count','complete_vehicle_count','partial_vehicle_count',
        'valid_cycle_comparisons','qualified_cycle_comparisons','observed_cycle_count','incomplete_cycle_count',
        'timing_eligible_prediction_count','qualified_prediction_count','unknown_qualification_prediction_count',
        'observed_wait_sample_count','complete_wait_sample_count','motion_unknown_sec','phase_unknown_sec',
        'motion_unknown_vehicle_count','phase_unknown_vehicle_count','unattributed_crossing_count','unattributed_stopped_sec')


def validate_budget(budget):
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<1:
        raise ValueError('Retention budget must be positive')
    return budget


class Retention:
    def __init__(self,budget,directory=None):
        validate_budget(budget)
        self.budget=budget;self.serial=0;self.acknowledged=0
        # Identity history of evicted rows, derived from the archive (see rebuilt).
        self.directory=directory;self.index=ArchiveIndex(directory);self.indexed=0
        self.totals={key:0 for key in COUNTS}
        self.wait_sum=0.;self.complete_sum=0.
        self.recent=deque(maxlen=budget)
        self.cycle_population={}
        self.archived_counts={key:0 for key in ('vehicles','comparisons','predictions','phase_events','observation_gaps')}
        self.visited_records=0;self.pending=None;self.progress={}

    def summary(self,rows,comparisons,method,predictions):
        rows=list(rows);comparisons=list(comparisons);predictions=list(predictions)
        self.visited_records=len(rows)+len(comparisons)+len(predictions)
        result=summarize(rows,comparisons,method,predictions)
        if self.archived_counts['vehicles']:
            count=self.totals['observed_vehicle_count']+len(rows)
            complete=self.totals['complete_vehicle_count']+result['complete_vehicle_count']
            complete_rows=[r for r in rows if r['completed'] and not r['left_censored'] and not r['right_censored'] and not r['quality_flags']]
            result['observed_mean_red_wait_sec']=round((self.wait_sum+sum(r['red_wait_sec'] for r in rows))/count,6) if count else None
            result['complete_mean_red_wait_sec']=round((self.complete_sum+sum(r['red_wait_sec'] for r in complete_rows))/complete,6) if complete else None
            result['observed_median_red_wait_sec']=result['observed_p95_red_wait_sec']=None
        for key in COUNTS:result[key]+=self.totals[key]
        for key,field in DURATIONS.items():result[key]=round(self.totals[key]+sum(r.get(field,0.) for r in rows),6)
        n=result['observed_vehicle_count'];valid=result['valid_cycle_comparisons']
        result['partial_vehicle_fraction']=result['partial_vehicle_count']/n if n else None
        result['qualified_comparison_fraction']=result['qualified_cycle_comparisons']/valid if valid else None
        result['quantiles']={'state':'unavailable' if self.archived_counts['vehicles'] else 'exact_live',
                             'reason':'Request exact quantiles for a durable generation' if self.archived_counts['vehicles'] else None}
        return result

    def prepare(self,engine,state):
        """Prepare a post-ack checkpoint without mutating unacknowledged evidence."""
        if self.pending is not None:return deepcopy(self.pending)
        self.serial+=1
        token={'serial':self.serial,'vehicles':{},'cycles':[], 'predictions':[], 'events':0,'gaps':0}
        archive=[]
        def add(kind,identity,value):archive.append({'kind':kind,'id':identity,'value':deepcopy(value)})
        cycles=engine.phase.cycles
        # An unfinished inference and a revisable hysteresis suffix retain their
        # cycle dependencies. Everything else outside the display window seals.
        protected={r['cycle_id'] for r in state['predictions'] if r['status']=='predicting'}
        for active in engine.waits.active.values():
            candidates=[active[k] for k in ('stop_candidate','move_candidate') if active[k] is not None]
            if candidates:
                protected.update(i['cycle_id'] for i in active['row']['intervals'] if i['start']>=min(candidates)-1e-8)
        drop={c['cycle_id'] for c in cycles[:-self.budget] if c['cycle_id'] not in protected}
        retained_cycles={c['cycle_id'] for c in cycles if c['cycle_id'] not in drop}
        if engine.phase.green:retained_cycles.add(engine.phase.green['cycle_id'])
        retained_cycles.add(engine.phase.cycle_id+1)
        for comparison in state['comparisons']:
            if comparison['cycle_id'] in drop:
                add('comparisons',comparison['cycle_id'],comparison);token['cycles'].append(comparison)
        state['comparisons']=[c for c in state['comparisons'] if c['cycle_id'] not in drop]
        current_rows={r['vehicle_id']:r for r in state['vehicles']}
        # recent display rows are already archived, not active checkpoint rows
        state['vehicles']=[r for r in state['vehicles'] if r['vehicle_id'] in engine.waits.records]
        states={v['row']['vehicle_id']:v for v in engine.waits.active.values()}
        for identity,row in engine.waits.records.items():
            snapshot=current_rows[identity]
            # Internal persistence retains full precision; public row rendering
            # rounds only after exact statistics have been derived.
            snapshot.update({k:v for k,v in row.items() if isinstance(v,float)})
            if row['completed']:
                add('vehicles',identity,snapshot)
                token['vehicles'][identity]={'final':deepcopy(snapshot)}
                state['vehicles'].remove(snapshot)
                continue
            active=states[identity]
            candidates=[active[k] for k in ('stop_candidate','move_candidate') if active[k] is not None]
            cutoff=min(candidates)-1e-8 if candidates else float('inf')
            count=0
            for interval in snapshot['intervals']:
                if interval['end']>cutoff:break
                count+=1
            old_cycles=[str(c) for c in snapshot['cycles'] if c not in retained_cycles and
                        not any(i['cycle_id']==c for i in snapshot['intervals'][count:])]
            if count:add('vehicle_intervals',identity,snapshot['intervals'][:count])
            # Membership and actual waits are separate: no zero wait is invented.
            if old_cycles:add('vehicle_cycles',identity,{'cycles':[int(c) for c in old_cycles],
                'wait_by_cycle':{c:snapshot['wait_by_cycle'][c] for c in old_cycles if c in snapshot['wait_by_cycle']}})
            token['vehicles'][identity]={'intervals':count,'cycles':old_cycles}
            snapshot['intervals']=snapshot['intervals'][count:]
            snapshot['cycles']=[c for c in snapshot['cycles'] if str(c) not in old_cycles]
            snapshot['wait_by_cycle']={c:v for c,v in snapshot['wait_by_cycle'].items() if c not in old_cycles}
        terminal=[r for r in state['predictions'] if r['status'] not in ('predicting','awaiting_green') and r['cycle_id'] not in retained_cycles and (engine.predictions.closed or r.get('completed_at') is not None )]
        evict={r['request_id'] for r in terminal[:-self.budget]}
        for row in terminal:
            if row['request_id'] in evict:add('predictions',row['request_id'],row);token['predictions'].append(row)
        state['predictions']=[r for r in state['predictions'] if r['request_id'] not in evict]
        for key,kind in (('phase_events','events'),('observation_gaps','gaps')):
            count=max(0,len(state.get(key,[]))-self.budget)
            token[kind]=count
            for index,row in enumerate(state.get(key,[])[:count]):
                add(key,self.archived_counts[key]+index,row)
            state[key]=state.get(key,[])[count:]
        state['_archive']=archive
        projected=deepcopy(self.totals);counts=deepcopy(self.archived_counts)
        wait_sum=self.wait_sum;complete_sum=self.complete_sum
        for change in token['vehicles'].values():
            if 'final' not in change:continue
            row=change['final'];stats=aggregate([row])
            for key in COUNTS:projected[key]+=stats[key]
            wait_sum+=row['red_wait_sec']
            if stats['complete_vehicle_count']:complete_sum+=row['red_wait_sec']
            counts['vehicles']+=1
        for kind in ('comparisons','predictions'):
            rows=token['cycles'] if kind=='comparisons' else token['predictions']
            stats=summarize([],rows if kind=='comparisons' else [],None,rows if kind=='predictions' else [])
            for key in COUNTS:projected[key]+=stats[key]
            counts[kind]+=len(rows)
        counts['phase_events']+=token['events'];counts['observation_gaps']+=token['gaps']
        state['retention']={'budget':self.budget,'archived_counts':counts,
                            'aggregates':projected,'wait_sum':wait_sum,'complete_sum':complete_sum}
        self.pending=deepcopy((state,token))
        return state,token

    def identity(self,base):
        """(rows so far, last row crossed) for an evicted native identity, else None."""
        value=self.index.get(b'v'+base.encode())
        return None if value is None else (value>>1,bool(value&1))

    def requested(self,request_id):
        """Whether this run archived and evicted the capture (no archive scan)."""
        return self.index.get(b'p'+request_id.encode('utf-8','surrogatepass')) is not None

    def _derive(self,kind,row):
        if kind=='vehicles':
            base,_,ordinal=row['vehicle_id'].partition(':segment-')
            segments=int(ordinal)+1 if ordinal else 1
            self.index.put(b'v'+base.encode(),segments*2+bool(row['crossed_stop_line']),keep=lambda old,new:old>>1>new>>1)
        elif kind=='predictions':
            self.index.put(b'p'+row['request_id'].encode('utf-8','surrogatepass'),1)

    def index_token(self,token):
        """Derive lookups before acknowledgment removes rows from memory."""
        if token['serial']<=self.indexed:return
        for change in token['vehicles'].values():
            if 'final' in change:self._derive('vehicles',change['final'])
        for row in token['predictions']:self._derive('predictions',row)
        self.indexed=token['serial']

    def rebuilt(self,records):
        """An equivalent index derived only from authoritative archive records."""
        copy=Retention(self.budget,self.directory)
        for record in records:copy._derive(record['kind'],record['value'])
        return copy

    def acknowledge(self,engine,token):
        if token['serial']<=self.acknowledged:return
        self.index_token(token)
        for identity,change in token['vehicles'].items():
            row=engine.waits.records.get(identity)
            if row is None:continue
            if 'final' in change:
                final=change['final'];stats=aggregate([final])
                for key in COUNTS:self.totals[key]+=stats[key]
                self.wait_sum+=final['red_wait_sec']
                if stats['complete_vehicle_count']:self.complete_sum+=final['red_wait_sec']
                self.archived_counts['vehicles']+=1
                recent=deepcopy(final);recent.pop('intervals',None)
                recent['history']=self.progress.pop(identity,{'archived_intervals':0,'archived_cycles':0})
                self.recent.append(recent)
                for cycle_id in final['cycles']:
                    population=self.cycle_population.setdefault(cycle_id,{'count':0,'wait':0.,'partial':0,'crossings':[]})
                    population['count']+=1
                    population['wait']+=final['wait_by_cycle'].get(str(cycle_id),{}).get('red_wait_sec',0.)
                    population['partial']+=bool(final['left_censored'] or final['right_censored'] or final['quality_flags'])
                    if final['crossed_stop_line'] and final['crossing_cycle']==cycle_id:
                        population['crossings'].append(final['completed_at'])
                del engine.waits.records[identity]
            else:
                progress=self.progress.setdefault(identity,{'archived_intervals':0,'archived_cycles':0})
                progress['archived_intervals']+=change['intervals'];progress['archived_cycles']+=len(change['cycles'])
                del row['intervals'][:change['intervals']]
                for cycle in change['cycles']:
                    row['wait_by_cycle'].pop(cycle,None)
                    if int(cycle) in row['cycles']:row['cycles'].remove(int(cycle))
        for identity,active in list(engine.waits.active.items()):
            if active['row']['vehicle_id'] not in engine.waits.records:
                if engine.waits.last_time-active.get('_last_seen',active['row']['last_seen'])>engine.policy.association_gap_sec:
                    del engine.waits.active[identity]
                    continue
                active['row']= {k:v for k,v in active['row'].items() if k not in ('intervals','wait_by_cycle','cycles')}
                active['previous']=None
                active['stop_candidate']=active['move_candidate']=None
        drop={c['cycle_id'] for c in token['cycles']}
        engine.phase.cycles[:]=[c for c in engine.phase.cycles if c['cycle_id'] not in drop]
        for comparison in token['cycles']:
            stats=summarize([], [comparison],None)
            for key in COUNTS:self.totals[key]+=stats[key]
            self.archived_counts['comparisons']+=1
        for cycle in list(self.cycle_population):
            if cycle in drop:self.cycle_population.pop(cycle)
        with engine.predictions.lock:
            for row in token['predictions']:
                engine.predictions.records.pop(row['request_id'],None)
                stats=summarize([],[],None,[row])
                for key in COUNTS:self.totals[key]+=stats[key]
                self.archived_counts['predictions']+=1
        del engine.phase.events[:token['events']];del engine.gaps[:token['gaps']]
        self.archived_counts['phase_events']+=token['events'];self.archived_counts['observation_gaps']+=token['gaps']
        # Only the phase observer's earliest revisable boundary can change
        # whether a recent crossing belongs inside a green interval.
        cycles={c['cycle_id']:c for c in engine.phase.cycles}
        if engine.phase.green:cycles[engine.phase.green['cycle_id']]=engine.phase.green
        for cycle_id,population in self.cycle_population.items():
            cycle=cycles.get(cycle_id)
            if not cycle:continue
            terminal='green_end' in cycle
            cutoff=cycle['green_end'] if terminal else (engine.phase.candidate_time
                if engine.phase.candidate!=engine.phase.phase else engine.phase.last_time)
            if cutoff is None:continue
            population['served']=population.get('served',0)+sum(cycle['green_start']<=t<cutoff for t in population['crossings'])
            population['crossings']=[] if terminal else [t for t in population['crossings'] if t>=cutoff]
        self.acknowledged=token['serial'];self.pending=None
