"""Opt-in delayed trajectory association; held evidence is never credited."""
from collections import deque
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.experiment.identity_contracts import IdentityResolution, ResolvedVehicle
from src.experiment.vehicle_identity import Entry, VehicleIdentityRegistry

def alias(item):
    return (item.native_epoch, item.native_id) if item.native_stable and item.native_id is not None else None


def dimensions(item):
    b = item.bbox
    return b[2]-b[0], b[3]-b[1]


def shape_jump(old, item):
    return any(max(a/b,b/a)>1.25 for a,b in zip(dimensions(old),dimensions(item)))


def reversal(entry, item):
    if len(entry.image_history)<2:
        return False
    (_,a),(_,b)=list(entry.image_history)[-2:]
    previous=np.subtract(b,a);step=np.subtract(item.point,b)
    scale=max(8.,dimensions(entry.detection)[1])
    return (np.linalg.norm(previous)>.03*scale and np.linalg.norm(step)>.03*scale
        and float(previous@step)<-.25*np.linalg.norm(previous)*np.linalg.norm(step))


class CandidateRegistry(VehicleIdentityRegistry):
    delayed = True
    priority = True
    def __init__(self, policy, max_gap_sec=.5, *, ground_trusted=True):
        super().__init__(policy, max_gap_sec, ground_trusted=ground_trusted)
        self.continuity_v4 = policy.association in ('delayed_trajectory_v4', 'delayed_trajectory_v5', 'delayed_trajectory_v6')
        self.lineage_v5 = policy.association in ('delayed_trajectory_v5', 'delayed_trajectory_v6')
        self.provisional_v6 = policy.association == 'delayed_trajectory_v6'
        self.window = 8
        self.holds = {}
        self.rejected_aliases = {}
        self.decisions = []

    def invalidate(self, reason):
        super().invalidate(reason)
        self.holds.clear()
        self.rejected_aliases.clear()
        self.decisions = []

    def retire(self, generation, identity, *, crossed=False):
        super().retire(generation, identity, crossed=crossed)
        self.holds.pop(identity, None)
        self.rejected_aliases.pop(identity, None)

    def _cost(self, entry, item, timestamp):
        if self.provisional_v6 and not entry.admitted:
            native = alias(item)
            if native is None or entry.first_native_reference != native:
                return None
        cost = super()._cost(entry, item, timestamp)
        if cost is None or not self.continuity_v4:
            return cost
        native = alias(item)
        if (not entry.admitted and native in entry.aliases
              and entry.sequence == self.last_sequence-1
              and entry.image_velocity is None):
            # A rejected native link cannot buy a wider motion gate by restarting
            # as a one-point tentative. Reuse v2's supported-motion bound.
            rejected = any(state['rejected'] and state['alias'] == native
                and identity in self.entries and self.entries[identity].admitted
                and timestamp-self.entries[identity].seen <= 1.
                for identity, state in self.holds.items())
            if self.lineage_v5 and not rejected:
                rejected = any(native in natives and identity in self.entries
                    and self.entries[identity].admitted
                    and timestamp-self.entries[identity].seen <= 1.
                    for identity, natives in self.rejected_aliases.items())
            if rejected:
                _, predicted, height, _ = self._geometry(entry, timestamp)
                if math.dist(predicted, item.point)/height > .3+.3*(timestamp-entry.seen):
                    return None
        return cost

    def _merged_continuity(self, entry, item, detections, timestamp):
        """Exceptional cross-native/release evidence only; no normal native guard."""
        if shape_jump(entry.detection, item):
            return True
        native = alias(item)
        def independent(other):
            return other is not item and alias(other) != native and alias(other) not in entry.aliases
        def covered(other):
            a, b = item.bbox, other.bbox
            intersection = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
            return intersection/((b[2]-b[0])*(b[3]-b[1])) >= .70
        if any(independent(other) and covered(other) for other in detections):
            return True
        return any(other.identity != entry.identity and timestamp-other.seen <= self.max_gap_sec
            and other.lifecycle not in ('EXITED', 'SERVED')
            and independent(other.detection) and covered(other.detection)
            for other in self.entries.values())

    def _update(self, entry, item, timestamp, sequence):
        if self.lineage_v5:
            # Supported evidence starts a new lineage, even for an old alias.
            # Keep rejection history separately for tentative-restart safety.
            hold = self.holds.pop(entry.identity, None)
            if hold is not None and hold['rejected']:
                self._reject_hold(entry, hold)
        recovered = sequence != entry.sequence+1 or timestamp-entry.seen > self.max_gap_sec
        if recovered:
            entry.history.clear()
            entry.image_history.clear()
        entry.image_history.append((timestamp, item.point))
        # A single sparse return has no new velocity evidence. Keep the last
        # measured image motion so intermittent boxes cannot imply stationarity.
        if len(entry.image_history) >= 2 and timestamp-entry.image_history[0][0] >= .05:
            times = np.asarray([t-entry.image_history[0][0] for t, _ in entry.image_history])
            times -= times.mean()
            points = np.asarray([p for _, p in entry.image_history])
            entry.image_velocity = tuple(times @ (points-points.mean(axis=0)) / (times @ times))
        if item.ground is not None:
            entry.history.append((timestamp, item.ground, item.point))
        else:
            entry.history.clear()
        valid = len(entry.history) >= 5 and entry.history[-1][0]-entry.history[0][0] >= .15
        speed = None
        if valid:
            origin,point,_=entry.history[0]
            st=stt=sx=sy=stx=sty=0.
            for sample,position,_ in entry.history:
                t=sample-origin;x=position[0]-point[0];y=position[1]-point[1]
                st+=t;stt+=t*t;sx+=x;sy+=y;stx+=t*x;sty+=t*y
            n=len(entry.history);denominator=n*stt-st*st
            speed=math.hypot((n*stx-st*sx)/denominator,(n*sty-st*sy)/denominator) if denominator>1e-12 else float('nan')
            valid = math.isfinite(speed) and speed <= 35
        entry.speed = speed if valid else None
        entry.observations += 1
        entry.admitted = entry.admitted or (entry.observations >= self.policy.confirmation_observations and timestamp-entry.born >= self.policy.confirmation_sec-1e-8)
        if item.appearance_reliable and item.appearance is not None:
            entry.appearance = item.appearance
            entry.trusted_detection=item
        if item.native_stable and item.native_id is not None:
            alias = (item.native_epoch, item.native_id)
            if entry.first_native_reference is None:entry.first_native_reference=alias
            if not entry.aliases or entry.aliases[-1] != alias:
                entry.aliases.append(alias)
        entry.exit_since = None if item.in_region else (timestamp if entry.exit_since is None else entry.exit_since)
        exit_pending = entry.exit_since is not None and timestamp-entry.exit_since < self.policy.exit_persistence_sec-1e-8
        if entry.lifecycle not in ('EXITED','SERVED'):
            entry.lifecycle = 'VISIBLE' if entry.admitted else 'TENTATIVE'
        entry.detection = item
        entry.seen = timestamp
        entry.sequence = sequence
        if not entry.admitted:
            return None
        return ResolvedVehicle(entry.identity, entry.generation, item, speed if valid else None,
            bool(valid and not exit_pending), recovered, exit_pending=exit_pending,
            birth_timestamp=entry.born,first_native_reference=entry.first_native_reference,association=self.policy.association)

    def _hold_state(self, entry, item):
        hold = self.holds.get(entry.identity)
        if self.lineage_v5 and hold is not None and hold['rejected']:
            self._reject_hold(entry, hold)
        if (self.lineage_v5 and hold is not None
                and ((hold['rejected'] and hold['alias'] != alias(item))
                     or hold.get('lineage', (entry.generation, entry.sequence))
                         != (entry.generation, entry.sequence))):
            self.holds.pop(entry.identity)
            return None
        return hold

    def _reject_hold(self, entry, hold, rejected_alias=None):
        hold['rejected'] = True
        if self.lineage_v5:
            # Same bound as the entry's native-alias history, tied to its lifetime.
            natives = self.rejected_aliases.setdefault(entry.identity, deque(maxlen=entry.aliases.maxlen))
            for native in (hold['alias'], rejected_alias):
                if native is None:
                    continue
                if native in natives:
                    natives.remove(native)
                natives.append(native)
        return 'reject'

    def _hold(self, entry, item, detections, timestamp, sequence):
        """Return hold/accept/reject; keep at most K points and entry-capacity states."""
        hold=self._hold_state(entry, item)
        if hold is None:
            hold={'samples':deque(maxlen=self.window),'start':sequence,'alias':alias(item),
                  'switched':False,'rejected':False}
            if self.lineage_v5:
                hold['lineage'] = (entry.generation, entry.sequence)
            self.holds[entry.identity]=hold
        if hold['rejected']:
            return 'reject'
        if hold['alias']!=alias(item):
            return self._reject_hold(entry, hold, alias(item))
        old,pred,h,gate=VehicleIdentityRegistry._geometry(self,entry,timestamp)
        # A different returned box on the old extrapolated path contradicts the link.
        for rival in detections:
            if rival is item or alias(rival)==alias(item):continue
            if math.dist(rival.point,pred)<.3*h and math.dist(rival.point,pred)<math.dist(item.point,pred):
                hold['switched']=True
        reliable=item.appearance_reliable and item.appearance is not None and entry.appearance is not None
        contradiction=(shape_jump(old,item) or reversal(entry,item) or hold['switched']
            or (reliable and math.dist(item.appearance,entry.appearance)/math.sqrt(2)>self.policy.appearance_threshold*.5))
        if contradiction:
            return self._reject_hold(entry, hold)
        hold['samples'].append((timestamp,item.point))
        count=len(hold['samples'])
        if count>=4:
            samples=list(entry.image_history)[-2:]+list(hold['samples'])
            times=np.array([t-samples[0][0] for t,p in samples]);points=np.array([p for t,p in samples])
            fit=np.polynomial.polynomial.polyfit(times,points,2)
            residual=float(np.max(np.linalg.norm(np.polynomial.polynomial.polyval(times,fit).T-points,axis=1)))/h
            if residual<=.10 and reliable:
                if self.continuity_v4 and self._merged_continuity(entry, item, detections, timestamp):
                    return self._reject_hold(entry, hold)
                return 'accept'
        if sequence-hold['start']+1>=self.window:
            return self._reject_hold(entry, hold)
        return 'hold'

    def resolve(self, detections, timestamp, sequence):
        if not math.isfinite(timestamp):raise ValueError('Identity timestamp must be finite')
        if self.last_sequence is not None and (sequence<=self.last_sequence or timestamp<=self.last_time):
            return IdentityResolution(duplicate=True)
        self.last_time,self.last_sequence=timestamp,sequence
        self.decisions=[]
        for identity,entry in list(self.entries.items()):
            ttl=self.policy.occlusion_retention_sec if entry.admitted else max(1.,self.policy.confirmation_sec*2)
            if timestamp-entry.seen>ttl:
                del self.entries[identity];self.holds.pop(identity,None)
                self.rejected_aliases.pop(identity, None)
        # Same duplicate threshold and ordering as v2.
        boxes=np.asarray([d.bbox for d in detections],dtype=float).reshape(-1,4)
        if len(boxes):
            intersection=np.prod(np.maximum(0.,np.minimum(boxes[:,None,2:],boxes[None,:,2:])-np.maximum(boxes[:,None,:2],boxes[None,:,:2])),axis=2)
            area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
            duplicates=intersection/np.maximum(1e-9,area[:,None]+area[None,:]-intersection)>.85
        kept=[]
        for i,item in enumerate(detections):
            if not kept or not duplicates[i,kept].any():kept.append(i)
        entries=sorted(self.entries.values(),key=lambda e:e.identity)
        costs=np.full((len(entries),len(kept)),1e6)
        if entries and kept:
            geometry=[self._geometry(e,timestamp) for e in entries]
            predicted=np.asarray([g[1] for g in geometry]);points=np.asarray([detections[i].point for i in kept])
            candidates=np.linalg.norm(predicted[:,None,:]-points[None,:,:],axis=2)/np.asarray([g[2] for g in geometry])[:,None]<=np.asarray([g[3] for g in geometry])[:,None]
            for row,col in np.argwhere(candidates):
                cost=self._cost(entries[row],detections[kept[col]],timestamp)
                if cost is not None:costs[row,col]=cost
        # Reuse v2's gated matrix. The native index is linear in retained aliases;
        # no extra scalar cost calls or Python pair scans on ordinary frames.
        own_by_alias = {}
        for row, entry in enumerate(entries) if self.delayed else ():
            if entry.admitted and entry.lifecycle not in ('EXITED', 'SERVED'):
                for native in entry.aliases:
                    previous = own_by_alias.get(native)
                    if previous is None or entry.seen > entries[previous].seen:
                        own_by_alias[native] = row
        blocked = set()
        forced = {}
        reserved = set()
        for col, index in enumerate(kept):
            item = detections[index]
            row = own_by_alias.get(alias(item))
            if row is None:
                continue
            entry = entries[row]
            state = self._hold_state(entry, item)
            if state is not None and state['rejected']:
                continue
            if state is not None or (costs[row, col] >= 1e6 and timestamp-entry.seen <= 1.):
                decision = self._hold(entry, item, detections, timestamp, sequence)
                self.decisions.append(dict(index=index, identity=entry.identity, kind=decision))
                if decision == 'accept':
                    (a, p), (b, q) = list(self.holds[entry.identity]['samples'])[-2:]
                    entry.image_velocity = tuple((q[k]-p[k])/(b-a) for k in (0, 1))
                    forced[index] = entry
                    reserved.add(row)
                    self.holds.pop(entry.identity, None)
                elif decision == 'hold':
                    blocked.add(index)
                    reserved.add(row)
        if reserved:
            costs[list(reserved), :] = 1e6
        if blocked or forced:
            costs[:, [c for c, i in enumerate(kept) if i in blocked or i in forced]] = 1e6
        # P2 only has work on contested columns. Preserve strict cheaper-than
        # ordering and recent/cheaper rivals, including genuine row ambiguity.
        for col in np.flatnonzero((costs < 1e6).sum(axis=0) > 1) if self.priority else ():
            native = alias(detections[kept[col]])
            valid = np.flatnonzero(costs[:, col] < 1e6)
            own = [r for r in valid if entries[r].admitted
                   and entries[r].sequence == sequence-1 and native in entries[r].aliases]
            for row in own:
                for other in valid:
                    entry = entries[other]
                    if other == row or costs[other, col] <= costs[row, col]:
                        continue
                    if timestamp-entry.seen > self.max_gap_sec or (not entry.admitted and native in entry.aliases):
                        costs[other, col] = 1e6
        matches=dict(forced)
        if costs.size:
            rows,cols=linear_sum_assignment(costs)
            for row,col in zip(rows,cols):
                cost=costs[row,col]
                if cost>=1e6:continue
                alternatives=np.concatenate((np.delete(costs[row],col),np.delete(costs[:,col],row)))
                if alternatives.size and alternatives.min()-cost<self.policy.ambiguity_margin:
                    entries[row].lifecycle='AMBIGUOUS'
                    for competitor in np.flatnonzero(costs[:,col]<cost+self.policy.ambiguity_margin):entries[competitor].lifecycle='AMBIGUOUS'
                    continue
                # Unsafe cross evidence still competes: deleting it from costs
                # could manufacture certainty for another track (V7 regression).
                item = detections[kept[col]]
                if (self.continuity_v4 and entries[row].admitted
                        and alias(item) not in entries[row].aliases
                        and self._merged_continuity(entries[row], item, detections, timestamp)):
                    continue
                matches[kept[col]]=entries[row]
        labels=[None]*len(detections);capacity=False
        for index in kept:
            if index in blocked:continue
            item=detections[index];entry=matches.get(index)
            if entry is None:
                col=kept.index(index)
                if costs.shape[0] and (costs[:,col]<1e6).any():continue
                pending=False
                for old in entries:
                    if not old.admitted or old.lifecycle in ('EXITED','SERVED') or timestamp-old.seen<=self.max_gap_sec:continue
                    if self.continuity_v4 and (self.holds.get(old.identity, {}).get('rejected', False)
                            or self.lineage_v5 and old.identity in self.rejected_aliases):continue
                    reference=old.detection
                    if math.dist(reference.point,item.point)>max(8.,dimensions(reference)[1])*.85:continue
                    comparable=item.appearance_reliable and item.appearance is not None and old.appearance is not None and len(item.appearance)==len(old.appearance)
                    if not comparable or math.dist(item.appearance,old.appearance)/math.sqrt(2)<=self.policy.appearance_threshold:
                        pending=True;break
                if pending or not item.in_region:continue
                if len(self.entries)>=self.policy.capacity:capacity=True;continue
                entry=Entry(self.next_id,self.generation,item,timestamp,timestamp,sequence,observations=0)
                self.next_id+=1;self.entries[entry.identity]=entry
            labels[index]=self._update(entry,item,timestamp,sequence)
        for entry in self.entries.values():
            if entry.seen<timestamp and entry.lifecycle not in ('EXITED','SERVED','AMBIGUOUS'):
                entry.lifecycle='OCCLUDED' if entry.admitted else 'TENTATIVE'
        return IdentityResolution(tuple(x for x in labels if x is not None),tuple(labels),tuple(i for i,x in enumerate(labels) if x is None),capacity_exhausted=capacity,
            ambiguous_logical_ids=tuple(e.identity for e in self.entries.values() if e.lifecycle=='AMBIGUOUS'))
