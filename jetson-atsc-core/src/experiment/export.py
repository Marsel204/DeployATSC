"""A checkpoint pins a prefix of the typed, append-only evidence archive.

Only this reader interprets the layout. schema_version=2 also existed before this
protocol; absent storage_protocol always means a read-only legacy snapshot.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
import uuid
import zipfile

from src.common.source_identity import public_data, discover_sources

PROTOCOL = 'typed-archive-v1'
TABLES = ('predictions', 'phase_events', 'vehicles', 'comparisons', 'observation_gaps')
IDS = {'predictions':'request_id', 'vehicles':'vehicle_id', 'comparisons':'cycle_id'}
# One explicitly rebuilt history, reused by page requests for the same pinned
# generation (each HTTP request opens a new reader). Committed bytes are immutable.
_PAGE_VIEW = {'key':None, 'view':None}
_PAGE_LOCK = threading.Lock()


def encoded(value):
    return json.dumps(value,allow_nan=False,sort_keys=True,separators=(',',':')).encode()


class ExperimentStore:
    def __init__(self, root, run_id, manifest):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid run identity')
        self.directory=Path(root)/run_id
        self.directory.mkdir(parents=True,exist_ok=False)
        self.manifest=dict(manifest,schema_version=2,storage_protocol=PROTOCOL,run_id=run_id,
                           created_utc=datetime.now(timezone.utc).isoformat())
        discover_sources(self.manifest)
        self.lock=threading.RLock()
        # Runs are discovered by manifest.json, so it is published last, after a
        # coherent generation 0. A failed creation is never a blocking run.
        try:
            with (self.directory/'archive.jsonl').open('xb') as stream:
                self._stage('archive_create_fsync',lambda:os.fsync(stream.fileno()))
            self._write('evidence.json',{'storage_protocol':PROTOCOL,'generation':0,'committed_bytes':0,
                                       'snapshot_revision':-1,'state':{'run_id':run_id}})
            self._write('manifest.json',self.manifest)
            self._stage('parent_directory_fsync',lambda:self._sync_directory(self.directory.parent))
        except BaseException:
            self._discard_creation()
            raise

    def _discard_creation(self):
        """Unpublish first; if that fails the run is still a coherent generation 0."""
        try:(self.directory/'manifest.json').unlink(missing_ok=True)
        except Exception:return
        try:
            for path in self.directory.iterdir():
                try:path.unlink()
                except Exception:pass
            self.directory.rmdir()
        except Exception:pass

    @classmethod
    def open(cls,directory):
        self=cls.__new__(cls)
        self.directory=Path(directory);self.lock=threading.RLock()
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        pinned=self.pin()  # validate before returning an apparently usable reader
        # Historical APIs may return a prediction/page without its source block.
        # Prepare manifest and checkpoint credentials before the first rendering.
        discover_sources(self.manifest)
        discover_sources(pinned['state'])
        return self

    def _stage(self,name,operation):
        """Named boundaries also permit deterministic before/after fault injection."""
        return operation()

    @staticmethod
    def _sync_directory(path):
        descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

    @staticmethod
    def _write_all(stream,data):
        view=memoryview(data)
        while view:
            count=stream.write(view)
            if not count:raise OSError('Short evidence write')
            view=view[count:]

    def _write(self,name,data):
        temporary=self.directory/(name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('xb',buffering=0) as stream:
                self._stage('checkpoint_write',lambda:self._write_all(stream,encoded(data)))
                self._stage('checkpoint_fsync',lambda:os.fsync(stream.fileno()))
            self._stage('checkpoint_replace',lambda:os.replace(temporary,self.directory/name))
            self._stage('directory_fsync',lambda:self._sync_directory(self.directory))
        finally:
            # Cleanup must never mask the original write/replace/fsync exception.
            try:temporary.unlink(missing_ok=True)
            except Exception:pass

    def pin(self):
        checkpoint=json.loads((self.directory/'evidence.json').read_text())
        if self.manifest.get('storage_protocol')!=PROTOCOL:
            if self.manifest.get('storage_protocol') not in (None,'legacy_snapshot'):
                raise ValueError('Unsupported evidence storage protocol')
            return {'storage_protocol':'legacy_snapshot','generation':None,'state':checkpoint}
        if checkpoint.get('storage_protocol')!=PROTOCOL:raise ValueError('Invalid evidence checkpoint protocol')
        boundary=checkpoint['committed_bytes']
        if boundary<0 or (self.directory/'archive.jsonl').stat().st_size<boundary:
            raise ValueError('Committed archive is incomplete')
        return checkpoint

    def persist(self,state):
        if self.manifest.get('storage_protocol')!=PROTOCOL:
            raise ValueError('Legacy evidence is read-only; start a new run')
        # Never invoke engine/session code under this lock. The supplied payload
        # is an immutable snapshot prepared by the caller before disk operations.
        payload=deepcopy(state)
        batch=payload.pop('_archive',[])
        operation_hash=hashlib.sha256(encoded({'state':payload,'archive':batch})).hexdigest()
        with self.lock:
            previous=self.pin()
            revision=payload.get('snapshot_revision',0)
            order=(revision,payload.get('commit_sequence',0))
            previous_order=(previous['snapshot_revision'],previous['state'].get('commit_sequence',0))
            if order<previous_order:raise ValueError('stale snapshot revision')
            if order==previous_order:
                if previous.get('operation_hash')!=operation_hash:
                    raise ValueError('stale or conflicting snapshot revision')
                # An earlier replace may have succeeded before directory fsync
                # failed. Confirm the same generation, without appending again.
                with (self.directory/'archive.jsonl').open('rb') as stream:os.fsync(stream.fileno())
                with (self.directory/'evidence.json').open('rb') as stream:os.fsync(stream.fileno())
                self._stage('directory_fsync',lambda:self._sync_directory(self.directory))
                return deepcopy(previous)
            boundary=previous['committed_bytes']
            data=b''.join(encoded(record)+b'\n' for record in batch)
            with (self.directory/'archive.jsonl').open('r+b',buffering=0) as stream:
                stream.truncate(boundary);stream.seek(boundary)
                self._stage('archive_write',lambda:self._write_all(stream,data))
                self._stage('archive_fsync',lambda:os.fsync(stream.fileno()))
            checkpoint={'storage_protocol':PROTOCOL,'generation':previous['generation']+1,
                        'committed_bytes':boundary+len(data),'snapshot_revision':revision,
                        'operation_hash':operation_hash,'state':payload}
            self._write('evidence.json',checkpoint)
            return deepcopy(checkpoint)

    def records(self,pinned=None):
        pinned=self.pin() if pinned is None else pinned
        if pinned['storage_protocol']!=PROTOCOL:return
        remaining=pinned['committed_bytes']
        with (self.directory/'archive.jsonl').open('rb') as stream:
            while remaining:
                line=stream.readline(remaining)
                if not line or not line.endswith(b'\n'):raise ValueError('Incomplete committed archive record')
                remaining-=len(line)
                yield json.loads(line)

    def read(self,pinned=None,*,raw=False):
        pinned=self.pin() if pinned is None else pinned
        state=deepcopy(pinned['state'])
        if pinned['storage_protocol']!=PROTOCOL:return state
        tables={key:{} for key in TABLES};fragments={}
        def add(kind,identity,value):
            if kind in ('vehicle_intervals','vehicle_cycles'):
                target=fragments.setdefault(identity,{'intervals':[], 'cycles':[], 'wait_by_cycle':{}})
                if kind=='vehicle_intervals':target['intervals'].extend(value)
                else:
                    # Earlier fragments keyed membership by cycle with a wait value.
                    cycles,waits=(value['cycles'],value['wait_by_cycle']) if 'cycles' in value else (list(value),value)
                    for cycle in cycles:
                        if int(cycle) not in target['cycles']:target['cycles'].append(int(cycle))
                    target['wait_by_cycle'].update(waits)
            elif kind in tables:tables[kind][str(identity)]=value
            else:raise ValueError('Unsupported archive record type: '+kind)
        for record in self.records(pinned):add(record['kind'],record['id'],record['value'])
        for key in TABLES:
            for index,row in enumerate(state.get(key,[])):
                identity=row.get(IDS.get(key,''),row.get('_archive_id',f'active-{index}'))
                tables[key][str(identity)]=row
            state[key]=list(tables[key].values())
        for row in state['vehicles']:
            fragment=fragments.get(row['vehicle_id'])
            if fragment:
                row['intervals']=fragment['intervals']+row.get('intervals',[])
                row['wait_by_cycle']={**fragment['wait_by_cycle'],**row.get('wait_by_cycle',{})}
                row['cycles']=sorted(set(fragment['cycles']+row.get('cycles',[])))
        state['vehicles'].sort(key=lambda r:(r.get('first_seen',0),r['vehicle_id']))
        state['comparisons'].sort(key=lambda r:r['cycle_id'])
        state['predictions'].sort(key=lambda r:(r.get('clicked_at',0),r.get('clicked_utc',''),r['request_id']))
        # Cycle partial/served populations depend on eventual vehicle outcomes,
        # including tracks which outlive many cycles. Rebuild on explicit reads.
        if state['comparisons'] and all('green_start' in c for c in state['comparisons']):
            by_cycle={}
            for row in state['vehicles']:
                for cycle_id in row.get('cycles',[]):by_cycle.setdefault(cycle_id,[]).append(row)
            for cycle in state['comparisons']:
                rows=by_cycle.get(cycle['cycle_id'],[])
                values=[r.get('wait_by_cycle',{}).get(str(cycle['cycle_id']),{}).get('red_wait_sec',0.) for r in rows]
                end=cycle.get('green_end',state.get('time',0))
                cycle.update(observed_vehicle_count=len(rows),
                    observed_mean_red_wait_sec=round(sum(values)/len(values),6) if values else None,
                    partial_vehicle_count=sum(bool(r.get('left_censored') or r.get('right_censored') or r.get('quality_flags')) for r in rows),
                    n_served=sum(bool(r.get('crossed_stop_line') and r.get('crossing_cycle')==cycle['cycle_id'] and
                                     cycle['green_start']<=r['completed_at']<end) for r in rows))
        if 'policy' in state:
            from src.experiment.summary import summarize
            state['summary']=summarize(state['vehicles'],state['comparisons'],state['policy'].get('los_method'),state['predictions'])
            state['los']=state['summary']['los']
        if 'lifecycle' in state:
            state['lifecycle'].update(durable_revision=pinned['snapshot_revision'],durable_generation=pinned['generation'],
                                       last_persisted_revision=pinned['snapshot_revision'])
        if not raw:
            for row in state['vehicles']:
                row.update({k:round(v,6) for k,v in row.items() if isinstance(v,float)})
        state['durability']={'protocol':PROTOCOL,'generation':pinned['generation'],
                             'durable_revision':pinned['snapshot_revision'],'committed_bytes':pinned['committed_bytes']}
        return state

    def page(self,kind,offset=0,limit=100,pinned=None):
        if kind not in TABLES:raise ValueError('Unknown evidence table')
        if offset<0 or not 1<=limit<=200:raise ValueError('Invalid pagination')
        pinned=self.pin() if pinned is None else pinned
        key=(str(self.directory.resolve()),pinned['generation'],pinned.get('committed_bytes'),pinned.get('snapshot_revision'))
        with _PAGE_LOCK:
            if _PAGE_VIEW['key']!=key:_PAGE_VIEW.update(key=key,view=self.read(pinned))
            rows=_PAGE_VIEW['view'].get(kind,[])
        return {'rows':deepcopy(rows[offset:offset+limit]),'total':len(rows),'offset':offset,
                'next_offset':offset+limit if offset+limit<len(rows) else None,
                'generation':pinned['generation'],'snapshot_revision':pinned.get('snapshot_revision')}

    def lookup(self,request_id,pinned=None):
        pinned=self.pin() if pinned is None else pinned
        for row in pinned['state'].get('predictions',[]):
            if row['request_id']==request_id:return deepcopy(row)
        result=None
        for record in self.records(pinned):
            if record['kind']=='predictions' and record['id']==request_id:result=record['value']
        return result

    def quantiles(self):
        pinned=self.pin();generation=pinned['generation']
        cached=getattr(self,'_quantile_cache',None)
        if cached is not None and cached['generation']==generation:return deepcopy(cached)
        from src.experiment.summary import _percentile
        values=[r.get('red_wait_sec',0.) for r in self.read(pinned,raw=True).get('vehicles',[])]
        result={'state':'exact','generation':generation,'snapshot_revision':pinned.get('snapshot_revision'),
                'sample_count':len(values),'median_red_wait_sec':_percentile(values,.5),'p95_red_wait_sec':_percentile(values,.95)}
        self._quantile_cache=result
        return deepcopy(result)

    def recover(self):
        """Finalize an interrupted NEW-protocol run; never resume measurements.

        The caller must own admission and establish that this run has no worker.
        Legacy folders are immutable and remain readable without this operation.
        """
        pinned=self.pin()
        if pinned['storage_protocol']!=PROTOCOL:raise ValueError('Legacy evidence is read-only')
        if self.recovery_status()['durable_finalized']:return self.recovery_status()
        state=deepcopy(pinned['state'])
        interrupted=self.recovery_status()['status']
        state['closed']=True;state['prediction_busy']=False
        state['snapshot_revision']=pinned['snapshot_revision']+1
        for row in state.get('predictions',[]):
            if row['status'] in ('predicting','awaiting_green'):
                row.update(status='unmatched',eligible=False,qualified=False,invalidation_reason='unclean_recovery')
                if row.get('qualification'):row['qualification'].update(state='diagnostic',reasons=['unclean_recovery'])
            if (row.get('camera_analysis') or {}).get('state')=='pending':
                row['camera_analysis']['state']='interrupted_unclean_recovery'
        for row in state.get('vehicles',[]):
            if not row.get('completed'):
                row.update(completed=True,completed_at=state.get('time'),right_censored=True,tracking_state='UNKNOWN')
                row.setdefault('quality_flags',[]).append('unclean_recovery')
        state.setdefault('lifecycle',{}).update(finalized=True,durable_finalized=True,resources_released=True,
            measurement_sealed=True,terminal_state='finalized',recovered_interrupted=True,recovered_from=interrupted)
        self.persist(state)
        return self.recovery_status()

    def recovery_status(self):
        pinned=self.pin();state=pinned['state'];lifecycle=state.get('lifecycle',{})
        finalized=bool(lifecycle.get('durable_finalized'))
        # Legacy snapshots carry no durable lifecycle; their outcome is not inferred.
        status=('legacy' if pinned['storage_protocol']!=PROTOCOL else 'finalized' if finalized
                else 'unclean' if not state.get('closed') else 'abandoned')
        result={'run_id':self.manifest.get('run_id'),'storage_protocol':pinned['storage_protocol'],
                'durable_revision':pinned.get('snapshot_revision'),'generation':pinned['generation'],
                'status':status,'durable_finalized':finalized}
        if lifecycle.get('recovered_interrupted'):result['recovered_from']=lifecycle.get('recovered_from','interrupted')
        return result

    @staticmethod
    def _csv(rows):
        stream=io.StringIO(newline='')
        fields=sorted({key for row in rows for key in row}) or ['run_id']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})
        return stream.getvalue()

    def export(self,state=None):
        if state is not None:self.persist(state)
        pinned=self.pin();evidence=public_data(self.read(pinned))
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr('manifest.json',json.dumps(public_data(self.manifest),indent=2,allow_nan=False))
            bundle.writestr('evidence.json',json.dumps(evidence,indent=2,allow_nan=False))
            for key in TABLES:bundle.writestr(key+'.csv',self._csv(evidence.get(key,[])))
        return output.getvalue()
