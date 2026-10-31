"""Bounded one-to-one physical evidence association; native IDs are only hints."""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.experiment.identity_contracts import IdentityResolution, ResolvedVehicle


def overlap(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1e-9, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


@dataclass
class Entry:
    identity: int
    generation: int
    detection: object
    born: float
    seen: float
    sequence: int
    observations: int = 1
    admitted: bool = False
    lifecycle: str = 'TENTATIVE'
    appearance: tuple | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    aliases: deque = field(default_factory=lambda: deque(maxlen=16))
    exit_since: float | None = None
    speed: float | None = None
    trusted_detection: object = None
    first_native_reference: tuple | None = None


class VehicleIdentityRegistry:
    def __init__(self, policy, max_gap_sec=.5, *, ground_trusted=True):
        self.policy = policy
        self.max_gap_sec = max_gap_sec
        self.ground_trusted=bool(ground_trusted)
        self.entries = {}
        self.generation = 0
        self.next_id = 1
        self.last_time = self.last_sequence = None

    def invalidate(self, reason):
        self.entries.clear()
        self.generation += 1
        self.last_time = self.last_sequence = None

    def retire(self, generation, identity, *, crossed=False):
        entry = self.entries.get(identity)
        if entry and entry.generation == generation:
            entry.lifecycle = 'SERVED' if crossed else 'EXITED'
            entry.history.clear()

    def _cost(self, entry, item, timestamp):
        if entry.lifecycle=='EXITED':return None
        old = entry.trusted_detection or entry.detection
        dt = timestamp-entry.seen
        height = max(8., old.bbox[3]-old.bbox[1])
        predicted = old.point
        if old is entry.detection and len(entry.history) >= 2 and entry.speed is not None and entry.speed > .5 and dt <= self.max_gap_sec:
            # Image displacement is independent of whether the physical scale is verified.
            previous = entry.history[-2]
            elapsed = entry.history[-1][0]-previous[0]
            if elapsed > 0:
                predicted = tuple(old.point[k]+(old.point[k]-previous[2][k])*dt/elapsed for k in (0, 1))
        distance = math.dist(predicted, item.point)/height
        gate = .65 + (min(dt, 2.)*.3 if entry.speed is not None and entry.speed > .5 else min(dt, 5.)*.04)
        if distance > gate:
            return None
        size = abs(math.log((item.bbox[2]-item.bbox[0])/(old.bbox[2]-old.bbox[0])))
        size += abs(math.log((item.bbox[3]-item.bbox[1])/(old.bbox[3]-old.bbox[1])))
        if size > 1.1:
            return None
        appearance = .25
        reliable = item.appearance_reliable and item.appearance is not None and entry.appearance is not None
        if reliable:
            if len(item.appearance) != len(entry.appearance):
                return None
            appearance = math.dist(item.appearance, entry.appearance)/math.sqrt(2)
            if appearance > self.policy.appearance_threshold:
                return None
        native = item.native_stable and item.native_id is not None
        alias = (item.native_epoch, item.native_id) if native else None
        same_alias = alias is not None and alias in entry.aliases
        if dt > self.max_gap_sec and not reliable and not same_alias:
            return None
        # Ground displacement adds a plausibility check without inventing a speed
        # across a hidden interval. Image-space evidence remains available unscaled.
        if self.ground_trusted and old.ground is not None and item.ground is not None:
            allowance = 2. + (entry.speed or 0.)*min(dt, 2.) + .5*dt
            if math.dist(old.ground, item.ground) > allowance:
                return None
        return .55*distance/gate + .2*min(1., size) + .25*appearance - (.02 if same_alias else 0)

    def _update(self, entry, item, timestamp, sequence):
        recovered = sequence != entry.sequence+1 or timestamp-entry.seen > self.max_gap_sec
        if recovered:
            entry.history.clear()
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
        if item.appearance_reliable and item.appearance is not None and (entry.appearance is None or not recovered):
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
            birth_timestamp=entry.born,first_native_reference=entry.first_native_reference)

    def resolve(self, detections, timestamp, sequence):
        if not math.isfinite(timestamp):
            raise ValueError('Identity timestamp must be finite')
        if self.last_sequence is not None and (sequence <= self.last_sequence or timestamp <= self.last_time):
            return IdentityResolution(duplicate=True)
        self.last_time, self.last_sequence = timestamp, sequence
        for identity, entry in list(self.entries.items()):
            ttl = self.policy.occlusion_retention_sec if entry.admitted else max(1., self.policy.confirmation_sec*2)
            if timestamp-entry.seen > ttl:
                del self.entries[identity]
        # Suppress duplicate admission downstream of the unchanged detector.
        boxes=np.asarray([d.bbox for d in detections],dtype=float).reshape(-1,4)
        if len(boxes):
            intersection=np.prod(np.maximum(0.,np.minimum(boxes[:,None,2:],boxes[None,:,2:])-
                np.maximum(boxes[:,None,:2],boxes[None,:,:2])),axis=2)
            area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
            duplicates=intersection/np.maximum(1e-9,area[:,None]+area[None,:]-intersection)>.85
        kept = []
        for i, item in enumerate(detections):
            if not kept or not duplicates[i,kept].any():
                kept.append(i)
        entries = sorted(self.entries.values(), key=lambda e: e.identity)
        costs = np.full((len(entries), len(kept)), 1e6)
        if entries and kept:
            references=[e.trusted_detection or e.detection for e in entries]
            predicted=np.asarray([d.point for d in references],dtype=float)
            for row,entry in enumerate(entries):
                dt=timestamp-entry.seen
                if references[row] is entry.detection and len(entry.history)>=2 and entry.speed is not None and entry.speed>.5 and dt<=self.max_gap_sec:
                    previous=entry.history[-2];elapsed=entry.history[-1][0]-previous[0]
                    if elapsed>0:predicted[row]+=(predicted[row]-np.asarray(previous[2]))*dt/elapsed
            points=np.asarray([detections[i].point for i in kept],dtype=float)
            heights=np.asarray([max(8.,d.bbox[3]-d.bbox[1]) for d in references])
            delta=np.asarray([timestamp-e.seen for e in entries])
            moving=np.asarray([e.speed is not None and e.speed>.5 for e in entries])
            gates=.65+np.where(moving,np.minimum(delta,2.)*.3,np.minimum(delta,5.)*.04)
            candidates=np.linalg.norm(predicted[:,None,:]-points[None,:,:],axis=2)/heights[:,None]<=gates[:,None]
            for row,col in np.argwhere(candidates):
                cost=self._cost(entries[row],detections[kept[col]],timestamp)
                if cost is not None:costs[row,col]=cost
        matches = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols):
                cost = costs[row, col]
                if cost >= 1e6:
                    continue
                alternatives = np.concatenate((np.delete(costs[row], col), np.delete(costs[:, col], row)))
                if alternatives.size and alternatives.min()-cost < self.policy.ambiguity_margin:
                    entries[row].lifecycle = 'AMBIGUOUS'
                    for competitor in np.flatnonzero(costs[:,col]<cost+self.policy.ambiguity_margin):
                        entries[competitor].lifecycle='AMBIGUOUS'
                    continue
                matches[kept[col]] = entries[row]
        labels = [None]*len(detections)
        capacity = False
        for index in kept:
            item = detections[index]
            entry = matches.get(index)
            if entry is None:
                col = kept.index(index)
                if costs.shape[0] and (costs[:, col] < 1e6).any():
                    continue  # Plausible but contested: no new stable timer.
                pending=False
                for old in entries:
                    if not old.admitted or old.lifecycle in ('EXITED','SERVED') or timestamp-old.seen<=self.max_gap_sec:continue
                    reference=old.trusted_detection or old.detection
                    if math.dist(reference.point,item.point)>max(8.,reference.bbox[3]-reference.bbox[1])*.85:continue
                    comparable=item.appearance_reliable and item.appearance is not None and old.appearance is not None and len(item.appearance)==len(old.appearance)
                    if not comparable or math.dist(item.appearance,old.appearance)/math.sqrt(2)<=self.policy.appearance_threshold:
                        pending=True;break
                if pending:continue
                if not item.in_region:
                    continue
                if len(self.entries) >= self.policy.capacity:
                    capacity = True
                    continue
                entry = Entry(self.next_id, self.generation, item, timestamp, timestamp, sequence,
                    observations=0)
                self.next_id += 1
                self.entries[entry.identity] = entry
            labels[index] = self._update(entry, item, timestamp, sequence)
        for entry in self.entries.values():
            if entry.seen < timestamp and entry.lifecycle not in ('EXITED','SERVED','AMBIGUOUS'):
                entry.lifecycle = 'OCCLUDED' if entry.admitted else 'TENTATIVE'
        return IdentityResolution(tuple(x for x in labels if x is not None), tuple(labels),
            tuple(i for i, x in enumerate(labels) if x is None), capacity_exhausted=capacity,
            ambiguous_logical_ids=tuple(e.identity for e in self.entries.values() if e.lifecycle=='AMBIGUOUS'))
