from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Mapping
import bisect
import math

from src.common.json_data import clone as deepcopy

from src.experiment.contracts import RunPolicy, vehicle_key


# Opt-in native fragment continuation (RunPolicy.fragment_continuation).
# Image-motion evidence is in pixels scaled by the vehicle's own box height.
HISTORY_SAMPLES = 8          # recent observations of one raw ID, gaps <= max_gap_sec
MAX_HISTORY_SAMPLES = 32     # cap when high FPS needs >8 samples to span MIN_FIT_SPAN_SEC
MIN_FIT_SAMPLES = 3
MIN_FIT_SPAN_SEC = .1
MIN_MOVING_HEIGHTS_PER_SEC = .5     # image speed that counts as motion evidence
CONTINUATION_BASE_HEIGHTS = .35
CONTINUATION_HEIGHTS_PER_SEC = .3
MAX_HEIGHT_RATIO = 1.5
# Without moving-motion evidence a newborn nearby is as likely a neighbour as
# the same vehicle, so only a flicker-length gap may be bridged.
STATIC_CONTINUATION_MAX_GAP_SEC = MIN_FIT_SPAN_SEC
# Bottom-edge occlusion on a stationary vehicle clips y2 upward while the top
# edge y1 stays pinned. That shrinkage must never fabricate upward velocity, and
# when unambiguous it anchors reappearance on the unoccluded top-center.
OCCLUSION_SHRINK_RATIO = .75
OCCLUSION_BOTTOM_SHARE = .7
OCCLUSION_TOP_DRIFT_HEIGHTS = .15
OCCLUSION_STATIC_TOP_SPEED = .35
OCCLUSION_MAX_GAP_SEC = .35
OCCLUSION_TOP_GATE_HEIGHTS = .25
OCCLUSION_BOTTOM_MARGIN_HEIGHTS = .15
OCCLUSION_MAX_RECOVERY_RATIO = 1.2
# A stationary vehicle whose bottom edge is clipped upward by a foreground
# occluder can cross the stop line while its unclipped rear stays behind it.
CROSSING_CLIP_SHRINK_RATIO = .85
CROSSING_RECOVERY_RATIO = .95
# Another observed box (any class, size or ID state) covering this much of the
# smaller of it and the newborn, with footpoints this close (in smaller-box
# heights), may be the same body or its neighbour: the link is ambiguous.
COMPETITOR_OVERLAP = .5
COMPETITOR_FOOTPOINT_HEIGHTS = .5
# Bounded work and memory: one row is continued at most this many times, and a
# frame with more newborn x lost pairs than this skips continuation entirely.
MAX_LINKS_PER_ROW = 8
MAX_PAIR_CHECKS = 4096


def _height(observation):
    height = observation.height
    if height is None and observation.bbox is not None:
        box = _raw_box(observation.bbox)
        height = None if box is None else box[3]-box[1]
    try:height = float(height)
    except (TypeError, ValueError, OverflowError):return None
    return height if math.isfinite(height) and height > 0 else None


def _fit(history):
    """Least-squares (vx, vy_bottom, vy_top) over timed (t, x, y, height) samples."""
    if len(history) < MIN_FIT_SAMPLES or history[-1][0]-history[0][0] < MIN_FIT_SPAN_SEC-1e-9:
        return None
    times = [h[0] for h in history]
    mean = sum(times)/len(times)
    denominator = sum((t-mean)**2 for t in times)
    if denominator <= 0:
        return None
    def slope(values):
        centre = sum(values)/len(values)
        return sum((t-mean)*(v-centre) for t, v in zip(times, values))/denominator
    vx = slope([h[1] for h in history])
    vy = slope([h[2] for h in history])
    vy_top = slope([h[2]-h[3] for h in history])
    return (vx, vy, vy_top) if all(map(math.isfinite, (vx, vy, vy_top))) else None


def _raw_box(box):
    if isinstance(box, (str, bytes, Mapping)):return None
    try:x1, y1, x2, y2 = (float(v) for v in box[:4])
    except (TypeError, ValueError, IndexError, OverflowError, KeyError):return None
    w, h = x2-x1, y2-y1
    return (x1, y1, x2, y2) if all(map(math.isfinite, (x1, y1, x2, y2, w, h, (x1+x2)/2, w*h))) and w > 0 and h > 0 else None


def _competitors(observations, unassigned):
    """(observation or None, footpoint, height, bbox) for every box of the frame,
    or None when any supplied box is unreadable (fail closed).

    An observation competes with its own box and, when it carries one, with
    the raw detection the tracker emitted it from (``source_bbox``): smoothing
    can move a tracked box out of a gate its detection is in. Both entries
    belong to the observation, so a newborn's own source never vetoes it.
    ``unassigned`` are the frame's actual detections the tracker did not emit:
    they can only veto a link, never become rows."""
    found = []
    for o in observations:
        box = None
        if o.bbox is not None:
            box = _raw_box(o.bbox)
            if box is None:return None
        try:px, py = float(o.point[0]), float(o.point[1])
        except (TypeError, ValueError, IndexError, OverflowError):return None
        if not (math.isfinite(px) and math.isfinite(py)):return None
        h = _height(o)
        if h is not None and not math.isfinite(py-h):return None
        found.append((o, (px, py), h, box))
        raw_source = getattr(o, "source_bbox", None)
        if raw_source is not None:
            source = _raw_box(raw_source)
            if source is None:return None
            if source != box:
                found.append((o, ((source[0]+source[2])/2, source[3]), source[3]-source[1], source))
    if isinstance(unassigned, (str, bytes, Mapping)):return None
    try:
        for raw in unassigned:
            box = _raw_box(raw)
            if box is None:return None
            found.append((None, ((box[0]+box[2])/2, box[3]), box[3]-box[1], box))
    except TypeError:
        return None
    return found


def _overlap_of_smaller(a, b):
    width = min(a[2], b[2])-max(a[0], b[0]); height = min(a[3], b[3])-max(a[1], b[1])
    if width <= 0 or height <= 0:return 0.
    smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return width*height/smaller if smaller > 0 else 0.


def _rear_err(y2, h):
    """Conservative float64 rearrangement slack between ``(y2-py)-0.7*(h-height)``
    and ``(y2-0.7*h)-(py-0.7*height)`` so dominance never drops cutoff candidates."""
    return max(1e-12, 16.0*math.ulp(max(abs(y2), abs(h), 1.0)))


def _dominates_coords(ax, ay1, ay2, ah, arear, aerr, bx, by1, by2, bh, brear, berr):
    """True when stationary baseline ``a`` provably covers ``b`` in height, rear
    bottom-share score (with rearrangement slack), and (x, y1) drift bounds."""
    if ax == bx and ay1 == by1 and ay2 == by2 and ah == bh:
        return True
    if ah < bh or arear-aerr < brear+berr:
        return False
    ra = OCCLUSION_TOP_DRIFT_HEIGHTS*ah
    rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
    return (ax-ra <= bx-rb and ax+ra >= bx+rb and
            ay1-ra <= by1-rb and ay1+ra >= by1+rb)


_KD_LEAF_MAX = 32


class _KDNode:
    __slots__ = ("axis", "split", "left", "right", "idxs", "count",
                 "min_x", "max_x", "min_y", "max_y", "min_h", "max_h",
                 "max_rear_lo", "min_rear_hi",
                 "min_x_hi", "max_x_hi", "min_x_lo", "max_x_lo",
                 "min_y_hi", "max_y_hi", "min_y_lo", "max_y_lo")

    def __init__(self, *, idxs=None, axis=0, split=0.0, left=None, right=None):
        self.axis = axis; self.split = split; self.left = left; self.right = right
        self.idxs = idxs
        self.count = 0
        self.min_x = self.min_y = self.min_h = math.inf
        self.max_x = self.max_y = self.max_h = -math.inf
        self.max_rear_lo = -math.inf; self.min_rear_hi = math.inf
        self.min_x_hi = self.min_x_lo = self.min_y_hi = self.min_y_lo = math.inf
        self.max_x_hi = self.max_x_lo = self.max_y_hi = self.max_y_lo = -math.inf

    def include_point(self, bx, by1, bh, brear, berr):
        r = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        x_hi = bx+r; x_lo = bx-r; y_hi = by1+r; y_lo = by1-r
        r_lo = brear-berr; r_hi = brear+berr
        if bx < self.min_x: self.min_x = bx
        if bx > self.max_x: self.max_x = bx
        if by1 < self.min_y: self.min_y = by1
        if by1 > self.max_y: self.max_y = by1
        if bh < self.min_h: self.min_h = bh
        if bh > self.max_h: self.max_h = bh
        if r_lo > self.max_rear_lo: self.max_rear_lo = r_lo
        if r_hi < self.min_rear_hi: self.min_rear_hi = r_hi
        if x_hi < self.min_x_hi: self.min_x_hi = x_hi
        if x_hi > self.max_x_hi: self.max_x_hi = x_hi
        if x_lo < self.min_x_lo: self.min_x_lo = x_lo
        if x_lo > self.max_x_lo: self.max_x_lo = x_lo
        if y_hi < self.min_y_hi: self.min_y_hi = y_hi
        if y_hi > self.max_y_hi: self.max_y_hi = y_hi
        if y_lo < self.min_y_lo: self.min_y_lo = y_lo
        if y_lo > self.max_y_lo: self.max_y_lo = y_lo
        self.count += 1

    def refresh_leaf(self, xs, y1s, hs, rears, errs):
        idxs = self.idxs
        if not idxs:
            self.count = 0
            self.min_x = self.min_y = self.min_h = math.inf
            self.max_x = self.max_y = self.max_h = -math.inf
            self.max_rear_lo = -math.inf; self.min_rear_hi = math.inf
            self.min_x_hi = self.min_x_lo = self.min_y_hi = self.min_y_lo = math.inf
            self.max_x_hi = self.max_x_lo = self.max_y_hi = self.max_y_lo = -math.inf
            return
        min_x = min_y = min_h = math.inf
        max_x = max_y = max_h = -math.inf
        max_rear_lo = -math.inf; min_rear_hi = math.inf
        min_x_hi = min_x_lo = min_y_hi = min_y_lo = math.inf
        max_x_hi = max_x_lo = max_y_hi = max_y_lo = -math.inf
        for i in idxs:
            bx = xs[i]; by1 = y1s[i]; bh = hs[i]
            r = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
            x_hi = bx+r; x_lo = bx-r; y_hi = by1+r; y_lo = by1-r
            r_lo = rears[i]-errs[i]; r_hi = rears[i]+errs[i]
            if bx < min_x: min_x = bx
            if bx > max_x: max_x = bx
            if by1 < min_y: min_y = by1
            if by1 > max_y: max_y = by1
            if bh < min_h: min_h = bh
            if bh > max_h: max_h = bh
            if r_lo > max_rear_lo: max_rear_lo = r_lo
            if r_hi < min_rear_hi: min_rear_hi = r_hi
            if x_hi < min_x_hi: min_x_hi = x_hi
            if x_hi > max_x_hi: max_x_hi = x_hi
            if x_lo < min_x_lo: min_x_lo = x_lo
            if x_lo > max_x_lo: max_x_lo = x_lo
            if y_hi < min_y_hi: min_y_hi = y_hi
            if y_hi > max_y_hi: max_y_hi = y_hi
            if y_lo < min_y_lo: min_y_lo = y_lo
            if y_lo > max_y_lo: max_y_lo = y_lo
        self.min_x = min_x; self.max_x = max_x
        self.min_y = min_y; self.max_y = max_y
        self.min_h = min_h; self.max_h = max_h
        self.max_rear_lo = max_rear_lo; self.min_rear_hi = min_rear_hi
        self.min_x_hi = min_x_hi; self.max_x_hi = max_x_hi
        self.min_x_lo = min_x_lo; self.max_x_lo = max_x_lo
        self.min_y_hi = min_y_hi; self.max_y_hi = max_y_hi
        self.min_y_lo = min_y_lo; self.max_y_lo = max_y_lo
        self.count = len(idxs)

    def refresh_internal(self):
        l, r = self.left, self.right
        self.min_x = l.min_x if l.min_x < r.min_x else r.min_x
        self.max_x = l.max_x if l.max_x > r.max_x else r.max_x
        self.min_y = l.min_y if l.min_y < r.min_y else r.min_y
        self.max_y = l.max_y if l.max_y > r.max_y else r.max_y
        self.min_h = l.min_h if l.min_h < r.min_h else r.min_h
        self.max_h = l.max_h if l.max_h > r.max_h else r.max_h
        self.max_rear_lo = l.max_rear_lo if l.max_rear_lo > r.max_rear_lo else r.max_rear_lo
        self.min_rear_hi = l.min_rear_hi if l.min_rear_hi < r.min_rear_hi else r.min_rear_hi
        self.min_x_hi = l.min_x_hi if l.min_x_hi < r.min_x_hi else r.min_x_hi
        self.max_x_hi = l.max_x_hi if l.max_x_hi > r.max_x_hi else r.max_x_hi
        self.min_x_lo = l.min_x_lo if l.min_x_lo < r.min_x_lo else r.min_x_lo
        self.max_x_lo = l.max_x_lo if l.max_x_lo > r.max_x_lo else r.max_x_lo
        self.min_y_hi = l.min_y_hi if l.min_y_hi < r.min_y_hi else r.min_y_hi
        self.max_y_hi = l.max_y_hi if l.max_y_hi > r.max_y_hi else r.max_y_hi
        self.min_y_lo = l.min_y_lo if l.min_y_lo < r.min_y_lo else r.min_y_lo
        self.max_y_lo = l.max_y_lo if l.max_y_lo > r.max_y_lo else r.max_y_lo
        self.count = l.count+r.count


def _build_kd(idxs, xs, y1s, hs, rears, errs):
    if len(idxs) <= _KD_LEAF_MAX:
        node = _KDNode(idxs=array("I", idxs))
        node.refresh_leaf(xs, y1s, hs, rears, errs)
        return node
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for i in idxs:
        bx = xs[i]; by1 = y1s[i]
        if bx < min_x: min_x = bx
        if bx > max_x: max_x = bx
        if by1 < min_y: min_y = by1
        if by1 > max_y: max_y = by1
    axis = 0 if (max_x-min_x) >= (max_y-min_y) else 1
    coords = xs if axis == 0 else y1s
    ordered = sorted(idxs, key=coords.__getitem__)
    mid = len(ordered)//2
    split = coords[ordered[mid]]
    left = _build_kd(ordered[:mid], xs, y1s, hs, rears, errs)
    right = _build_kd(ordered[mid:], xs, y1s, hs, rears, errs)
    node = _KDNode(axis=axis, split=split, left=left, right=right)
    node.refresh_internal()
    return node


def _collect_idxs(node, out):
    if node is None:
        return
    if node.idxs is not None:
        out.extend(node.idxs)
        return
    _collect_idxs(node.left, out)
    _collect_idxs(node.right, out)


_HBLOCK_MAX = 128
_HBLOCK_MERGE_TRIGGER = 32
_HBLOCK_MERGE_MAX = 96


class _HBlock:
    __slots__ = ("h", "y2", "rear", "err", "cnt", "max_rear_hi")

    def __init__(self, h=None, y2=None, rear=None, err=None, cnt=None, max_rear_hi=-math.inf):
        self.h = array("d") if h is None else h
        self.y2 = array("d") if y2 is None else y2
        self.rear = array("d") if rear is None else rear
        self.err = array("d") if err is None else err
        self.cnt = array("I") if cnt is None else cnt
        self.max_rear_hi = max_rear_hi

    def refresh_max_rear_hi(self):
        m = -math.inf
        rear = self.rear; err = self.err
        for i in range(len(rear)):
            v = rear[i]+err[i]
            if v > m: m = v
        self.max_rear_hi = m


class _ClipBaselines:
    """Lossless compact-array indexed stationary clip baselines for one active track.

    - Baselines are stored in contiguous ``array('d')`` columns (no per-entry
      boxed Python tuples or floats), giving minimal memory and O(1) teardown.
    - A scapegoat-balanced 2D (x, y1) KD-tree carrying height, rear, and
      drift-gate envelopes answers drift pruning (O(1) on stationary frames and
      O(1) on whole-track departure), dominance/duplicate checks, and live
      eviction of provably dominated entries.
    - A 2-level blocked sorted ``(h, y2)`` index (``_HBlock`` chunks of <= 128
      unique refcounted groups) supports O(log B) incremental inserts and
      deletes with zero rebuilds on partial drift pruning, plus O(1) straight-
      line crossing shortcuts.
    """
    __slots__ = ("_x", "_y1", "_y2", "_h", "_rear", "_err", "_free", "count",
                 "_root", "_recent_q", "_recent_s",
                 "_blocks", "_blk_max_h", "_blk_max_y2")

    def __init__(self):
        self._x = array("d")
        self._y1 = array("d")
        self._y2 = array("d")
        self._h = array("d")
        self._rear = array("d")
        self._err = array("d")
        self._free = array("I")
        self.count = 0
        self._root = None
        self._recent_q = deque()
        self._recent_s = set()
        self._blocks = []
        self._blk_max_h = array("d")
        self._blk_max_y2 = array("d")

    def __bool__(self):
        return self.count > 0

    def __len__(self):
        return self.count

    def clear(self):
        self._x = array("d"); self._y1 = array("d"); self._y2 = array("d")
        self._h = array("d"); self._rear = array("d"); self._err = array("d")
        self._free = array("I")
        self.count = 0
        self._root = None
        self._recent_q.clear(); self._recent_s.clear()
        self._blocks.clear()
        self._blk_max_h = array("d"); self._blk_max_y2 = array("d")

    def _find_block(self, h, y2):
        blocks = self._blocks
        max_h = self._blk_max_h; max_y2 = self._blk_max_y2
        b_idx = bisect.bisect_left(max_h, h)
        n = len(blocks)
        while b_idx < n and max_h[b_idx] == h and max_y2[b_idx] < y2:
            b_idx += 1
        return b_idx

    def _blocks_add(self, h, y2, rear, err):
        blocks = self._blocks
        max_h = self._blk_max_h; max_y2 = self._blk_max_y2
        r_hi = rear+err
        if not blocks:
            blk = _HBlock(array("d", (h,)), array("d", (y2,)), array("d", (rear,)),
                          array("d", (err,)), array("I", (1,)), r_hi)
            blocks.append(blk); max_h.append(h); max_y2.append(y2)
            return
        b_idx = self._find_block(h, y2)
        if b_idx == len(blocks):
            b_idx -= 1
        blk = blocks[b_idx]
        bh = blk.h; by2 = blk.y2
        i = bisect.bisect_left(bh, h)
        n = len(bh)
        while i < n and bh[i] == h:
            if by2[i] == y2:
                blk.cnt[i] += 1
                return
            if by2[i] > y2:
                break
            i += 1
        bh.insert(i, h); by2.insert(i, y2)
        blk.rear.insert(i, rear); blk.err.insert(i, err); blk.cnt.insert(i, 1)
        if r_hi > blk.max_rear_hi:
            blk.max_rear_hi = r_hi
        if i == len(bh)-1:
            max_h[b_idx] = h; max_y2[b_idx] = y2
        if len(bh) > _HBLOCK_MAX:
            mid = len(bh)//2
            right = _HBlock(bh[mid:], by2[mid:], blk.rear[mid:], blk.err[mid:], blk.cnt[mid:])
            right.refresh_max_rear_hi()
            del bh[mid:]; del by2[mid:]; del blk.rear[mid:]; del blk.err[mid:]; del blk.cnt[mid:]
            blk.refresh_max_rear_hi()
            max_h[b_idx] = bh[-1]; max_y2[b_idx] = by2[-1]
            blocks.insert(b_idx+1, right)
            max_h.insert(b_idx+1, right.h[-1])
            max_y2.insert(b_idx+1, right.y2[-1])

    def _blocks_remove(self, h, y2):
        blocks = self._blocks
        if not blocks:
            return
        max_h = self._blk_max_h; max_y2 = self._blk_max_y2
        b_idx = self._find_block(h, y2)
        if b_idx >= len(blocks):
            return
        blk = blocks[b_idx]
        bh = blk.h; by2 = blk.y2
        i = bisect.bisect_left(bh, h)
        n = len(bh)
        while i < n and bh[i] == h and by2[i] < y2:
            i += 1
        if i >= n or bh[i] != h or by2[i] != y2:
            return
        c = blk.cnt[i]-1
        if c > 0:
            blk.cnt[i] = c
            return
        removed_hi = blk.rear[i]+blk.err[i]
        del bh[i]; del by2[i]; del blk.rear[i]; del blk.err[i]; del blk.cnt[i]
        if not bh:
            del blocks[b_idx]; del max_h[b_idx]; del max_y2[b_idx]
            return
        if removed_hi >= blk.max_rear_hi:
            blk.refresh_max_rear_hi()
        max_h[b_idx] = bh[-1]; max_y2[b_idx] = by2[-1]
        if len(bh) < _HBLOCK_MERGE_TRIGGER and len(blocks) > 1:
            if b_idx+1 < len(blocks) and len(bh)+len(blocks[b_idx+1].h) <= _HBLOCK_MERGE_MAX:
                nxt = blocks[b_idx+1]
                bh.extend(nxt.h); by2.extend(nxt.y2)
                blk.rear.extend(nxt.rear); blk.err.extend(nxt.err); blk.cnt.extend(nxt.cnt)
                if nxt.max_rear_hi > blk.max_rear_hi: blk.max_rear_hi = nxt.max_rear_hi
                max_h[b_idx] = bh[-1]; max_y2[b_idx] = by2[-1]
                del blocks[b_idx+1]; del max_h[b_idx+1]; del max_y2[b_idx+1]
            elif b_idx > 0 and len(blocks[b_idx-1].h)+len(bh) <= _HBLOCK_MERGE_MAX:
                prv = blocks[b_idx-1]
                prv.h.extend(bh); prv.y2.extend(by2)
                prv.rear.extend(blk.rear); prv.err.extend(blk.err); prv.cnt.extend(blk.cnt)
                if blk.max_rear_hi > prv.max_rear_hi: prv.max_rear_hi = blk.max_rear_hi
                max_h[b_idx-1] = prv.h[-1]; max_y2[b_idx-1] = prv.y2[-1]
                del blocks[b_idx]; del max_h[b_idx]; del max_y2[b_idx]

    def _alloc_slot(self, bx, by1, by2, bh, brear, berr):
        if self._free:
            i = self._free.pop()
            self._x[i] = bx; self._y1[i] = by1; self._y2[i] = by2
            self._h[i] = bh; self._rear[i] = brear; self._err[i] = berr
        else:
            i = len(self._x)
            self._x.append(bx); self._y1.append(by1); self._y2.append(by2)
            self._h.append(bh); self._rear.append(brear); self._err.append(berr)
        self.count += 1
        return i

    def _free_slot(self, i):
        if self._recent_s:
            self._recent_s.discard((self._x[i], self._y1[i], self._y2[i], self._h[i]))
        self._blocks_remove(self._h[i], self._y2[i])
        self._free.append(i)
        self.count -= 1

    def _compact_slots(self):
        if self.count == 0:
            self.clear()
            return
        idxs = []
        _collect_idxs(self._root, idxs)
        nx = array("d"); ny1 = array("d"); ny2 = array("d")
        nh = array("d"); nrear = array("d"); nerr = array("d")
        ox, oy1, oy2, oh, orear, oerr = self._x, self._y1, self._y2, self._h, self._rear, self._err
        for i in idxs:
            nx.append(ox[i]); ny1.append(oy1[i]); ny2.append(oy2[i])
            nh.append(oh[i]); nrear.append(orear[i]); nerr.append(oerr[i])
        self._x, self._y1, self._y2, self._h, self._rear, self._err = nx, ny1, ny2, nh, nrear, nerr
        self._free = array("I")
        self.count = len(nx)
        self._root = _build_kd(range(self.count), nx, ny1, nh, nrear, nerr)

    def _free_subtree(self, node):
        if node.idxs is not None:
            for i in node.idxs:
                self._free_slot(i)
            return
        self._free_subtree(node.left)
        self._free_subtree(node.right)

    def _prune_node(self, node, px, y1):
        if node.min_x_hi >= px and node.max_x_lo <= px and node.min_y_hi >= y1 and node.max_y_lo <= y1:
            return node, False
        if node.max_x_hi < px or node.min_x_lo > px or node.max_y_hi < y1 or node.min_y_lo > y1:
            self._free_subtree(node)
            return None, True
        if node.idxs is not None:
            xs, y1s, hs = self._x, self._y1, self._h
            kept = array("I")
            removed = False
            for i in node.idxs:
                bx = xs[i]; by1 = y1s[i]; r = OCCLUSION_TOP_DRIFT_HEIGHTS*hs[i]
                if bx-r <= px <= bx+r and by1-r <= y1 <= by1+r:
                    kept.append(i)
                else:
                    self._free_slot(i)
                    removed = True
            if not removed:
                return node, False
            if not kept:
                return None, True
            node.idxs = kept
            node.refresh_leaf(xs, y1s, hs, self._rear, self._err)
            return node, True
        new_left, cl = self._prune_node(node.left, px, y1)
        new_right, cr = self._prune_node(node.right, px, y1)
        if not (cl or cr):
            return node, False
        if new_left is None:
            return new_right, True
        if new_right is None:
            return new_left, True
        node.left, node.right = new_left, new_right
        node.refresh_internal()
        return node, True

    def prune(self, px, y1):
        root = self._root
        if root is None:
            return False
        if root.min_x_hi >= px and root.max_x_lo <= px and root.min_y_hi >= y1 and root.max_y_lo <= y1:
            return True
        if root.max_x_hi < px or root.min_x_lo > px or root.max_y_hi < y1 or root.min_y_lo > y1:
            self.clear()
            return False
        self._root, _ = self._prune_node(root, px, y1)
        if self._root is None:
            self.clear()
            return False
        if len(self._x) > 4*max(self.count, 64):
            self._compact_slots()
        return True

    def _is_covered(self, node, bx, by1, by2, bh, brear, berr, rb, bx_lo, bx_hi, by_lo, by_hi, brear_hi):
        if node is None or node.max_h < bh:
            return False
        can_dominate = (node.max_rear_lo >= brear_hi and
                        node.min_x_lo <= bx_lo and node.max_x_hi >= bx_hi and
                        node.min_y_lo <= by_lo and node.max_y_hi >= by_hi)
        if can_dominate:
            reach = OCCLUSION_TOP_DRIFT_HEIGHTS*(node.max_h-bh)+max(1e-9, 16.0*math.ulp(max(abs(bx), abs(by1), node.max_h, 1.0)))
            if bx+reach < node.min_x or bx-reach > node.max_x or by1+reach < node.min_y or by1-reach > node.max_y:
                return False
        else:
            # No strict dominator exists in this subtree; only an exact duplicate can match.
            if bx < node.min_x or bx > node.max_x or by1 < node.min_y or by1 > node.max_y or bh < node.min_h:
                return False
        if node.idxs is not None:
            xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
            if not can_dominate:
                for i in node.idxs:
                    if xs[i] == bx and y1s[i] == by1 and y2s[i] == by2 and hs[i] == bh:
                        return True
                return False
            rears, errs = self._rear, self._err
            for i in node.idxs:
                if _dominates_coords(xs[i], y1s[i], y2s[i], hs[i], rears[i], errs[i],
                                     bx, by1, by2, bh, brear, berr):
                    return True
            return False
        return (self._is_covered(node.left, bx, by1, by2, bh, brear, berr, rb, bx_lo, bx_hi, by_lo, by_hi, brear_hi) or
                self._is_covered(node.right, bx, by1, by2, bh, brear, berr, rb, bx_lo, bx_hi, by_lo, by_hi, brear_hi))

    def covered(self, b):
        bx, by1, by2, bh, brear, berr = b
        if (bx, by1, by2, bh) in self._recent_s:
            return True
        rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        return self._is_covered(self._root, bx, by1, by2, bh, brear, berr,
                                rb, bx-rb, bx+rb, by1-rb, by1+rb, brear+berr)

    def _evict_dominated(self, node, ax, ay1, ay2, ah, arear, aerr, ax_lo, ax_hi, ay_lo, ay_hi, arear_lo):
        # Strict positive rearrangement slack guarantees any b != a dominated by a has b.h < ah.
        if (node is None or node.min_h >= ah or node.min_rear_hi > arear_lo or
                node.max_x_lo < ax_lo or node.min_x_hi > ax_hi or
                node.max_y_lo < ay_lo or node.min_y_hi > ay_hi):
            return node, False
        reach = OCCLUSION_TOP_DRIFT_HEIGHTS*(ah-node.min_h)+max(1e-9, 16.0*math.ulp(max(abs(ax), abs(ay1), ah, 1.0)))
        if ax+reach < node.min_x or ax-reach > node.max_x or ay1+reach < node.min_y or ay1-reach > node.max_y:
            return node, False
        if node.idxs is not None:
            xs, y1s, y2s, hs, rears, errs = self._x, self._y1, self._y2, self._h, self._rear, self._err
            kept = array("I")
            removed = False
            for i in node.idxs:
                if _dominates_coords(ax, ay1, ay2, ah, arear, aerr,
                                     xs[i], y1s[i], y2s[i], hs[i], rears[i], errs[i]):
                    self._free_slot(i)
                    removed = True
                else:
                    kept.append(i)
            if not removed:
                return node, False
            if not kept:
                return None, True
            node.idxs = kept
            node.refresh_leaf(xs, y1s, hs, rears, errs)
            return node, True
        new_left, cl = self._evict_dominated(node.left, ax, ay1, ay2, ah, arear, aerr, ax_lo, ax_hi, ay_lo, ay_hi, arear_lo)
        new_right, cr = self._evict_dominated(node.right, ax, ay1, ay2, ah, arear, aerr, ax_lo, ax_hi, ay_lo, ay_hi, arear_lo)
        if not (cl or cr):
            return node, False
        if new_left is None:
            return new_right, True
        if new_right is None:
            return new_left, True
        node.left, node.right = new_left, new_right
        node.refresh_internal()
        return node, True

    def _kd_insert(self, idx, bx, by1, bh, brear, berr):
        if self._root is None:
            node = _KDNode(idxs=array("I", (idx,)))
            node.include_point(bx, by1, bh, brear, berr)
            self._root = node
            return
        path = []
        node = self._root
        while node.idxs is None:
            node.include_point(bx, by1, bh, brear, berr)
            go_left = (bx if node.axis == 0 else by1) < node.split
            path.append((node, go_left))
            node = node.left if go_left else node.right
        node.idxs.append(idx)
        node.include_point(bx, by1, bh, brear, berr)
        if len(node.idxs) <= _KD_LEAF_MAX:
            return
        xs, y1s, hs, rears, errs = self._x, self._y1, self._h, self._rear, self._err
        split_node = _build_kd(node.idxs, xs, y1s, hs, rears, errs)
        if not path:
            self._root = split_node
            return
        parent, was_left = path[-1]
        if was_left: parent.left = split_node
        else: parent.right = split_node
        for depth_i, (anc, _) in enumerate(path):
            if anc.count >= 64 and 4*max(anc.left.count, anc.right.count) > 3*anc.count:
                sub = []
                _collect_idxs(anc, sub)
                rebuilt = _build_kd(sub, xs, y1s, hs, rears, errs)
                if depth_i == 0:
                    self._root = rebuilt
                else:
                    gp, gp_left = path[depth_i-1]
                    if gp_left: gp.left = rebuilt
                    else: gp.right = rebuilt
                return

    def add(self, b):
        bx, by1, by2, bh, brear, berr = b
        key = (bx, by1, by2, bh)
        if key in self._recent_s:
            return
        rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        bx_lo, bx_hi, by_lo, by_hi = bx-rb, bx+rb, by1-rb, by1+rb
        if self._is_covered(self._root, bx, by1, by2, bh, brear, berr,
                            rb, bx_lo, bx_hi, by_lo, by_hi, brear+berr):
            return
        if self._root is not None:
            self._root, _ = self._evict_dominated(self._root, bx, by1, by2, bh, brear, berr,
                                                  bx_lo, bx_hi, by_lo, by_hi, brear-berr)
        slot = self._alloc_slot(bx, by1, by2, bh, brear, berr)
        self._recent_s.add(key)
        self._recent_q.append(key)
        if len(self._recent_q) > MAX_HISTORY_SAMPLES:
            self._recent_s.discard(self._recent_q.popleft())
        self._blocks_add(bh, by2, brear, berr)
        self._kd_insert(slot, bx, by1, bh, brear, berr)
        if len(self._x) > 4*max(self.count, 64):
            self._compact_slots()

    def best_for_crossing(self, h_min, rear_floor, height, y1, px, eligible, side, best_key):
        """Return highest qualifying ``(h_base, y2_base)`` beating ``best_key`` (or ``best_key``
        unchanged) without rebuilding on partial drift pruning or scanning shorter heights
        after the tallest eligible baseline clears a straight stop line."""
        blocks = self._blocks
        if not blocks:
            return best_key
        h_floor = h_min-1e-9
        for b_idx in range(len(blocks)-1, -1, -1):
            blk = blocks[b_idx]
            top_h = blk.h[-1]
            if top_h < h_floor:
                break
            if best_key is not None and (top_h, blk.y2[-1]) <= best_key:
                break
            if blk.max_rear_hi < rear_floor:
                continue
            top_y2 = y1+top_h
            top_behind = side((px, top_y2)) < 0 if (top_h >= height and math.isfinite(top_y2)) else None
            if top_behind is False:
                return best_key
            bh = blk.h; by2 = blk.y2; brear = blk.rear; berr = blk.err
            last_i = len(bh)-1
            for i in range(last_i, -1, -1):
                h_base = bh[i]
                if h_base < h_floor:
                    return best_key
                y2_base = by2[i]
                if best_key is not None and (h_base, y2_base) <= best_key:
                    return best_key
                if brear[i]+berr[i] < rear_floor:
                    continue
                if not eligible(y2_base, h_base):
                    continue
                if i == last_i and top_behind is not None:
                    behind = top_behind
                else:
                    unclipped_y2 = y1+h_base
                    if not math.isfinite(unclipped_y2):
                        continue
                    behind = side((px, unclipped_y2)) < 0
                if behind:
                    return (h_base, y2_base)
                if h_base >= height:
                    return best_key
        return best_key


class VehicleWaits:
    """Integrate observed intervals. Missing observations are never imputed.

    Opt-in native fragment continuation (``RunPolicy.fragment_continuation``):
    a never-seen native ID continues a vehicle row lost <= max_gap_sec earlier
    only when the pair is mutually unique on position alone, the class and box
    size agree, and no other observed box competes. The interval across the
    handoff is unknown coverage (no wait, no crossing, no motion history), and
    every raw ID keeps its own segment and crossing history.

    ``unassigned`` boxes (``observe``) are the frame's other actual detections
    that the tracker withheld; they only veto links. ``None`` declares that
    evidence unavailable, and then no link is made in that frame.
    """
    fragment_continuation = True
    def __init__(self, run_id, policy: RunPolicy, side):
        self.run_id = run_id
        self.policy = policy
        self.side = side
        self.records = {}; self.work = 0
        self.active = {}
        self.last_sequence = None
        self.first_sequence = None
        self.last_time = None
        # Derived lookup of identities whose rows were archived and evicted.
        self.archived = None
        # Raw IDs that handed their row to a continuation: {identity: {"row_id", "segments"}}.
        # Pruned once the row is evicted; the archive index answers after that.
        self.handed_off = {}
        # Stable raw IDs recently seen outside the region (never "newborn").
        self.outside_seen = {}
        self.continuation_skipped_frames = 0
        # Frames with a continuation candidate but no complete same-frame detections.
        self.continuation_unavailable_frames = 0
        # Persistent identities record per-interval tracking continuity for
        # acquisition collection; legacy exports keep their historical schema.
        self.record_tracking = False

    def _flag(self, row, flag):
        if flag not in row["quality_flags"]: row["quality_flags"].append(flag)

    def _account(self, row, interval, multiplier=1):
        duration = interval["end"]-interval["start"]
        if interval["phase"] == "UNKNOWN":
            row["phase_unknown_sec"] += multiplier*duration
        if not interval["valid"]:
            row["unknown_sec"] += multiplier*duration
        elif interval["stopped"]:
            row["total_stopped_sec"] += multiplier*duration
            if interval["phase"] == "RED": row["red_wait_sec"] += multiplier*duration
            if interval["cycle_id"] is None:
                row["unattributed_stopped_sec"] += multiplier*duration
                return
            cycle = row["wait_by_cycle"].setdefault(str(interval["cycle_id"]), {"red_wait_sec":0., "total_stopped_sec":0.})
            cycle["total_stopped_sec"] += multiplier*duration
            if interval["phase"] == "RED":cycle["red_wait_sec"] += multiplier*duration

    def _revise_since(self, row, timestamp, stopped):
        for interval in reversed(row["intervals"]):
            if interval["start"] < timestamp-1e-8: break
            if interval["valid"] and interval["stopped"] != stopped:
                self._account(row, interval, -1)
                interval["stopped"] = stopped
                self._account(row, interval)

    def _new(self, observation, timestamp, sequence, segment=0):
        identity = (observation.epoch, observation.local_id)
        base = vehicle_key(self.run_id, *identity)
        # The identity's own row count, as in the full-history oracle.
        key = base if segment == 0 else f"{base}:segment-{segment}"
        row = {"vehicle_id": key, "local_id": observation.local_id,
            "camera": "acquisition", "tracker_epoch": observation.epoch,
            "class_name": observation.class_name, "first_seen": timestamp,
            "last_seen": timestamp, "red_wait_sec": 0., "total_stopped_sec": 0.,
            "unknown_sec": 0., "phase_unknown_sec":0., "unattributed_stopped_sec":0.,
            "cycles": [], "quality_flags": [], "intervals": [],
            "wait_by_cycle": {},
            "left_censored": segment > 0 or sequence == self.first_sequence or (observation.velocity_valid and observation.speed_mps <= self.policy.stopped_speed_mps),
            "right_censored": True, "completed": False, "crossed_stop_line": False,
            "crossing_cycle": None, "crossing_at":None,"crossing_phase":None,
            "crossing_direction":None,"crossing_evidence":None,
            "completed_at": None, "tracking_state": "UNKNOWN"}
        self.records[key] = row
        state = {"row": row, "previous": None, "stopped": False,
                 "stop_candidate": None, "move_candidate": None, "segments": segment+1}
        self.active[identity] = state
        return state

    def _restore(self, identity, timestamp):
        """Memory forgot an archived identity; its derived history still counts."""
        base = vehicle_key(self.run_id, *identity)
        found = self.archived(base)
        if found is None:return None
        segments, crossed = found
        # segments == 0: the raw ID only ever continued another row (an alias).
        row = {"vehicle_id": base if segments == 1 else f"{base}:segment-{segments-1}" if segments else f"{base}:alias",
               "tracker_epoch": identity[0], "local_id": identity[1],
               "completed": True, "crossed_stop_line": crossed, "last_seen": timestamp}
        state = {"row": row, "previous": None, "stopped": False, "stop_candidate": None,
                 "move_candidate": None, "segments": segments, "_last_seen": timestamp}
        self.active[identity] = state
        return state

    @staticmethod
    def _valid_box_geometry(observation):
        if observation.bbox is not None and _raw_box(observation.bbox) is None:
            return None
        height = _height(observation)
        if height is None:
            return None
        try:px, py = float(observation.point[0]), float(observation.point[1])
        except (TypeError, ValueError, IndexError, OverflowError, KeyError):return None
        if not all(map(math.isfinite, (px, py, py-height))):
            return None
        return px, py, height

    def _step_continuous(self, prev, timestamp, sequence):
        return bool(prev is not None and not prev.get("handoff") and
                    prev["sequence"] == sequence-1 and
                    0 < timestamp-prev["time"] <= self.policy.max_gap_sec+1e-9)

    @staticmethod
    def _clear_clip_state(state):
        state["pending_crossing"] = state["clip_baselines"] = None
        if "clip_history" in state:
            state["clip_history"].clear()

    def _record_history(self, state, prev_obs, observation, timestamp, consecutive):
        history = state.setdefault("history", deque(maxlen=MAX_HISTORY_SAMPLES))
        clip_history = state.setdefault("clip_history", deque(maxlen=MAX_HISTORY_SAMPLES))
        recent = (prev_obs is not None and not prev_obs.get("handoff") and
                  0 < timestamp-prev_obs["time"] <= self.policy.max_gap_sec+1e-9)
        height = _height(observation)
        try:px, py = float(observation.point[0]), float(observation.point[1])
        except (TypeError, ValueError, IndexError, OverflowError, KeyError):px = py = float("nan")
        valid_sample = height is not None and all(map(math.isfinite, (px, py, py-height)))
        if not recent or not valid_sample:history.clear()
        if valid_sample:
            history.append((timestamp, px, py, height))
            while len(history) > HISTORY_SAMPLES and history[-1][0]-history[1][0] >= MIN_FIT_SPAN_SEC-1e-9:
                history.popleft()
        clip_geom = self._valid_box_geometry(observation)
        if not consecutive or clip_geom is None:
            clip_history.clear()
            if clip_geom is None:
                self._clear_clip_state(state)
        if clip_geom is not None:
            clip_history.append((timestamp, clip_geom[0], clip_geom[1], clip_geom[2]))
            while len(clip_history) > HISTORY_SAMPLES and clip_history[-1][0]-clip_history[1][0] >= MIN_FIT_SPAN_SEC-1e-9:
                clip_history.popleft()

    @staticmethod
    def _corroborated_pairs(earlier):
        out = []
        for a, b in zip(earlier, earlier[1:]):
            y2 = a[2] if a[2] <= b[2] else b[2]
            h = a[3] if a[3] <= b[3] else b[3]
            out.append((b[1], y2-h, y2, h, y2-OCCLUSION_BOTTOM_SHARE*h, _rear_err(y2, h)))
        return out

    @staticmethod
    def _prune_clip_baselines(state, px, y1):
        saved = state.get("clip_baselines")
        if saved is not None and not saved.prune(px, y1):
            state["clip_baselines"] = None

    def _refresh_clip_baselines(self, state, observation, timestamp, prev_obs):
        """Record corroborated stationary baselines into the indexed set without
        scanning accumulated anchors on stationary frames."""
        history = state.get("clip_history")
        if not history or len(history) < 2 or prev_obs is None or history[-1][0] != prev_obs["time"]:
            return
        geom = self._valid_box_geometry(observation)
        if geom is None:
            return
        px, py, height = geom
        y1 = py-height
        earlier = list(history)
        pairs = list(dict.fromkeys(self._corroborated_pairs(earlier)))
        saved = state.get("clip_baselines")
        if saved is not None and all(saved.covered(p) for p in pairs):
            return
        samples = earlier+[(timestamp, px, py, height)]
        motion = _fit(samples)
        if motion is None:
            return
        y1_vals = [s[2]-s[3] for s in samples]
        x_vals = [s[1] for s in samples]
        y1_span = max(y1_vals)-min(y1_vals)
        x_span = max(x_vals)-min(x_vals)
        top_speed = math.hypot(motion[0], motion[2])
        valid = []
        for b in pairs:
            r = OCCLUSION_TOP_DRIFT_HEIGHTS*b[3]
            if (y1_span <= r and x_span <= r and
                    top_speed < OCCLUSION_STATIC_TOP_SPEED*b[3] and
                    b[0]-r <= px <= b[0]+r and b[1]-r <= y1 <= b[1]+r):
                valid.append(b)
        if valid:
            if saved is None:
                saved = _ClipBaselines()
                state["clip_baselines"] = saved
            for b in valid:
                saved.add(b)

    def _bottom_clipped_crossing(self, state, observation, timestamp, prev_obs):
        """Return ``(h_base, pair_to_save_or_None)`` when a stop-line crossing is
        explained purely by bottom-edge occlusion on a stationary vehicle, else None."""
        history = state.get("clip_history")
        if not history or len(history) < 2 or prev_obs is None or history[-1][0] != prev_obs["time"]:
            return None
        geom = self._valid_box_geometry(observation)
        if geom is None:
            return None
        px, py, height = geom
        y1 = py-height
        earlier = list(history)
        samples = earlier+[(timestamp, px, py, height)]
        motion = _fit(samples)
        if motion is None:
            return None
        y1_vals = [s[2]-s[3] for s in samples]
        x_vals = [s[1] for s in samples]
        y1_span = max(y1_vals)-min(y1_vals)
        x_span = max(x_vals)-min(x_vals)
        top_speed = math.hypot(motion[0], motion[2])
        best_key = None
        best_pair = None
        def eligible(y2_base, h_base):
            return (height <= CROSSING_CLIP_SHRINK_RATIO*h_base+1e-9 and
                    (y2_base-py) >= OCCLUSION_BOTTOM_SHARE*(h_base-height)-1e-9 and
                    y1_span <= OCCLUSION_TOP_DRIFT_HEIGHTS*h_base and
                    x_span <= OCCLUSION_TOP_DRIFT_HEIGHTS*h_base and
                    top_speed < OCCLUSION_STATIC_TOP_SPEED*h_base)
        for b in self._corroborated_pairs(earlier):
            y2_base, h_base = b[2], b[3]
            if not eligible(y2_base, h_base):
                continue
            unclipped_y2 = y1+h_base
            if not math.isfinite(unclipped_y2):
                continue
            # Orientation-agnostic counterfactual: only suppress when restoring the
            # pre-occlusion height places the vehicle's rear back behind the line.
            if self.side((px, unclipped_y2)) < 0:
                key = (h_base, y2_base)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = b
        saved = state.get("clip_baselines")
        if saved is not None:
            h_min = max(
                (height-1e-9)/CROSSING_CLIP_SHRINK_RATIO,
                y1_span/OCCLUSION_TOP_DRIFT_HEIGHTS,
                x_span/OCCLUSION_TOP_DRIFT_HEIGHTS,
                top_speed/OCCLUSION_STATIC_TOP_SPEED,
            )
            rear_floor = py-OCCLUSION_BOTTOM_SHARE*height-1e-9-_rear_err(py, height)
            saved_key = saved.best_for_crossing(h_min, rear_floor, height, y1, px, eligible, self.side, best_key)
            if saved_key is not None and (best_key is None or saved_key > best_key):
                best_key = saved_key
                best_pair = None
        return (best_key[0], best_pair) if best_key is not None else None

    def _pending_crossing_confirmed(self, pending, observation):
        """True once a clipped track's unclipped rear also clears the stop line."""
        geom = self._valid_box_geometry(observation)
        if geom is None:
            return False
        px, py, height = geom
        unclipped_y2 = py-height+pending["h_base"]
        if not math.isfinite(unclipped_y2):
            return False
        return bool(self.side(observation.point) >= 0 and self.side((px, unclipped_y2)) >= 0)

    def _evaluate_crossing(self, state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                           valid_motion, continuing, keep_pending):
        row = state["row"]
        geom = self._valid_box_geometry(observation)
        if continuing:
            if geom is None:
                self._clear_clip_state(state)
            else:
                self._prune_clip_baselines(state, geom[0], geom[1]-geom[2])
        pending = state.get("pending_crossing")
        if pending is not None:
            if self.side(observation.point) < 0:
                pending["from"] = {"time": timestamp, "sequence": sequence,
                    "phase": str(phase), "cycle": cycle_id, "valid_motion": valid_motion,
                    "point": observation.point}
                if geom is not None and geom[2] >= CROSSING_RECOVERY_RATIO*pending["h_base"]:
                    pending["recovered"] = pending.get("recovered", 0)+1
                    if pending["recovered"] >= 2:
                        state["pending_crossing"] = None
                else:
                    pending["recovered"] = 0
                return False
            pending["recovered"] = 0
            if self._pending_crossing_confirmed(pending, observation):
                self._cross(row, dict(pending["from"], phase=prev_obs["phase"]), observation, phase, timestamp, cycle_id)
                self._clear_clip_state(state)
                return True
            return False
        if self.side(prev_obs["point"]) < 0 <= self.side(observation.point):
            clipped = self._bottom_clipped_crossing(state, observation, timestamp, prev_obs) if continuing else None
            if clipped is not None:
                h_base, pair = clipped
                if pair is not None:
                    saved = state.get("clip_baselines")
                    if saved is None:
                        saved = _ClipBaselines()
                        state["clip_baselines"] = saved
                    saved.add(pair)
                if keep_pending:
                    state["pending_crossing"] = {"from": prev_obs, "h_base": h_base, "recovered": 0}
                return False
            self._cross(row, prev_obs, observation, phase, timestamp, cycle_id)
            self._clear_clip_state(state)
            return True
        if continuing:
            self._refresh_clip_baselines(state, observation, timestamp, prev_obs)
        return False

    def _cross(self, row, previous, observation, phase, timestamp, cycle_id):
        crossing_phase = str(phase) if previous["phase"] != "UNKNOWN" else "UNKNOWN"
        attributed = cycle_id if crossing_phase != "UNKNOWN" else None
        if not row["completed"]:self.work+=1
        row.update(crossed_stop_line=True,crossing_cycle=attributed,
                   crossing_at=timestamp,crossing_phase=crossing_phase,crossing_direction=1,
                   crossing_evidence={"from_point":previous["point"],"to_point":observation.point,
                                      "from_time":previous["time"],"to_time":timestamp,
                                      "from_side":self.side(previous["point"]),"to_side":self.side(observation.point)},
                   completed=True,completed_at=timestamp,right_censored=False,tracking_state="SERVED")
        if attributed is None:self._flag(row,"crossing_phase_unknown")

    def _clear_native_exit(self, state):
        if self.policy.exit_persistence_sec > 0:
            state["exit_candidate"] = state["exit_last"] = None

    def observe(self, observations, phase, timestamp, sequence, cycle_id, unassigned=None):
        if sequence == self.last_sequence or (self.last_time is not None and timestamp <= self.last_time): return
        if self.first_sequence is None: self.first_sequence = sequence
        self.last_sequence = sequence; self.last_time = timestamp
        if phase == "UNKNOWN":cycle_id = None
        continuing = self.fragment_continuation and self.policy.fragment_continuation
        native_exit = self.policy.exit_persistence_sec > 0
        if continuing:
            # Read more than once (prepasses, accounting): a one-shot iterator is materialized once.
            observations = observations if isinstance(observations, (list, tuple)) else list(observations)
            # A sighting older than the association timeout no longer makes an ID "not newborn".
            if self.outside_seen:
                self.outside_seen = {k: t for k, t in self.outside_seen.items()
                                     if timestamp-t <= self.policy.association_gap_sec}
            # A handed-off ID seen again contests the live link before any row of this
            # frame completes (e.g. the continuing ID crossing the line): order-independent.
            if self.handed_off:
                for observation in observations:
                    if observation.local_id is None or not observation.stable_id:continue
                    identity = (observation.epoch, observation.local_id)
                    if identity in self.handed_off:
                        self._contest(self.handed_off[identity]["row_id"], identity, timestamp)
            self._continue_fragments(observations, timestamp, unassigned)
        present = set()
        for observation in observations:
            if observation.local_id is None or not observation.stable_id: continue
            identity = (observation.epoch, observation.local_id)
            present.add(identity)
            state = self.active.get(identity)
            segment = 0; seen_before = False
            if state is None and identity in self.handed_off:
                handed = self.handed_off[identity]
                self._contest(handed["row_id"], identity, timestamp)
                if not observation.in_region:continue
                # The raw ID is back as its own vehicle: its own next segment.
                del self.handed_off[identity]
                segment = handed["segments"]; seen_before = True
            elif state is None and observation.in_region and self.archived is not None:
                state = self._restore(identity, timestamp)
            if state and state["row"]["completed"]:
                if state["row"]["crossed_stop_line"]:
                    state["_last_seen"]=timestamp
                    continue
                segment = self._own_segments(identity, state); seen_before = True
                state = None
            if not observation.in_region:
                if state is None and continuing:self.outside_seen[identity] = timestamp
                if state and not state["row"]["completed"]:
                    row=state["row"];previous=state["previous"]
                    in_native_exit = native_exit and state.get("exit_candidate") is not None
                    prev_obs = state.get("exit_last") if in_native_exit else previous
                    step_continuous = (self._step_continuous(prev_obs, timestamp, sequence) if native_exit else
                        bool(previous is not None and not previous.get("handoff") and
                             previous["sequence"] == sequence-1 and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec))
                    crossed_now = False
                    if step_continuous:
                        crossed_now = self._evaluate_crossing(
                            state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                            False, continuing, keep_pending=native_exit)
                    else:
                        self._clear_clip_state(state)
                    if crossed_now:
                        self._clear_native_exit(state)
                    elif not native_exit:
                        self._clear_clip_state(state)
                        row.update(completed=True,completed_at=timestamp,tracking_state="EXITED",right_censored=False);self.work+=1
                    else:
                        if not (in_native_exit and step_continuous):
                            state["exit_candidate"] = timestamp
                        state["exit_last"] = {"time": timestamp, "sequence": sequence,
                            "phase": str(phase), "cycle": cycle_id, "valid_motion": False,
                            "point": observation.point}
                        if continuing:
                            geom = self._valid_box_geometry(observation)
                            if geom is not None:
                                state["last_valid_box"] = (timestamp, geom[0], geom[1], geom[2])
                            self._record_history(state, prev_obs, observation, timestamp, step_continuous)
                        state["stop_candidate"] = state["move_candidate"] = None
                        if timestamp-state["exit_candidate"] >= self.policy.exit_persistence_sec-1e-8:
                            row.update(completed=True,completed_at=state["exit_candidate"],tracking_state="EXITED",right_censored=False);self.work+=1
                            self._clear_clip_state(state)
                            self._clear_native_exit(state)
                        else:
                            row["tracking_state"] = "EXIT_PENDING"
                continue
            if state is None:
                state = self._new(observation, timestamp, sequence, segment)
                if seen_before:state["row"]["left_censored"] = True
            was_exit_pending = native_exit and state.get("exit_candidate") is not None
            exit_last = state.get("exit_last") if was_exit_pending else None
            if was_exit_pending:
                self._clear_native_exit(state)
            row = state["row"]; previous = state["previous"]
            prev_obs = exit_last if was_exit_pending else previous
            valid_motion = observation.velocity_valid and observation.speed_mps is not None
            # Wait accounting is continuous only across consecutive in-ROI frames (never across handoffs or outside dips).
            continuous = (not was_exit_pending and previous is not None and not previous.get("handoff") and
                          previous["sequence"] == sequence-1 and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec)
            crossing_continuous = self._step_continuous(prev_obs, timestamp, sequence) if was_exit_pending else continuous
            if previous is not None:
                interval = {"start": previous["time"], "end": timestamp,
                            "phase": previous["phase"] if continuous else "UNKNOWN",
                            "cycle_id": previous["cycle"] if continuous else None,
                            "valid": bool(continuous and previous["valid_motion"] and valid_motion),
                            "stopped": state["stopped"]}
                # Same identity seen in consecutive frames: no tracking break.
                if self.record_tracking: interval["tracked"] = bool(continuous)
                row["intervals"].append(interval); self.work+=1; self._account(row, interval)
                if not continuous:
                    self._flag(row, "observation_gap")
                    state["stop_candidate"] = state["move_candidate"] = None
                if crossing_continuous:
                    self._evaluate_crossing(
                        state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                        valid_motion, continuing, keep_pending=True)
                else:
                    self._clear_clip_state(state)
            if valid_motion:
                speed = observation.speed_mps
                if not state["stopped"]:
                    state["move_candidate"] = None
                    if speed <= self.policy.stopped_speed_mps:
                        if state["stop_candidate"] is None: state["stop_candidate"] = timestamp
                        if timestamp-state["stop_candidate"] >= self.policy.stop_persistence_sec-1e-8:
                            self._revise_since(row, state["stop_candidate"], True)
                            state["stopped"] = True; state["stop_candidate"] = None
                    else: state["stop_candidate"] = None
                else:
                    state["stop_candidate"] = None
                    if speed >= self.policy.moving_speed_mps:
                        if state["move_candidate"] is None: state["move_candidate"] = timestamp
                        if timestamp-state["move_candidate"] >= self.policy.stop_persistence_sec-1e-8:
                            self._revise_since(row, state["move_candidate"], False)
                            state["stopped"] = False; state["move_candidate"] = None
                    else: state["move_candidate"] = None
                if not row["completed"]: row["tracking_state"] = "STOPPED" if state["stopped"] else "MOVING"
            else:
                self._flag(row, "velocity_unknown")
                state["stop_candidate"] = state["move_candidate"] = None
                if not row["completed"]: row["tracking_state"] = "UNKNOWN"
            if phase == "UNKNOWN": self._flag(row, "phase_unknown")
            if cycle_id is not None and cycle_id not in row["cycles"]: row["cycles"].append(cycle_id)
            row["last_seen"] = timestamp
            if continuing:
                geom = self._valid_box_geometry(observation)
                if geom is not None:
                    state["last_valid_box"] = (timestamp, geom[0], geom[1], geom[2])
                self._record_history(state, prev_obs, observation, timestamp, crossing_continuous)
            state["previous"] = {"time": timestamp, "sequence": sequence,
                "phase": str(phase), "cycle": cycle_id, "valid_motion": valid_motion,
                "point": observation.point}
        for identity, state in list(self.active.items()):
            row = state["row"]
            if row["vehicle_id"] not in self.records and identity not in present and timestamp-state.get("_last_seen",row["last_seen"])>self.policy.association_gap_sec:
                del self.active[identity]
                continue
            if identity not in present and not row["completed"]:
                row["tracking_state"] = "LOST"
                self._clear_clip_state(state)
                exit_last = state.get("exit_last") if native_exit else None
                last_observed = max(row["last_seen"], exit_last["time"]) if exit_last is not None else row["last_seen"]
                if timestamp-last_observed > self.policy.association_gap_sec:
                    row.update(completed=True, completed_at=timestamp);self.work+=1
                    self._flag(row, "tracking_lost")
                    self._clear_native_exit(state)

    def _own_segments(self, identity, state):
        """Rows this raw ID itself started; an alias of another row started none."""
        row = state["row"]
        return state["segments"] if identity == (row.get("tracker_epoch"), row.get("local_id")) else 0

    def _contest(self, row_id, identity, timestamp):
        """A handed-off raw ID is seen again: the live link is contradicted.

        Only an unfinished row is flagged; a finished row may already be archived."""
        row = self.records.get(row_id)
        if row is None or row["completed"]:return
        self._flag(row, "identity_link_contested")
        for link in row.get("identity_links", []):
            if (link["from"]["epoch"], link["from"]["local_id"]) == identity:link.setdefault("contested_at", timestamp)

    def release(self):
        """Drop handoff references to evicted rows (the archive index answers for them)."""
        if self.handed_off:
            self.handed_off = {k: v for k, v in self.handed_off.items() if v["row_id"] in self.records}

    def physical(self, identity):
        """The vehicle row a raw (epoch, native ID) currently drives, if any."""
        state = self.active.get(identity)
        if state is None or state["row"].get("vehicle_id") not in self.records:return None
        row = state["row"]
        return {"vehicle_id": row["vehicle_id"], "local_id": row["local_id"],
                "tracker_epoch": row["tracker_epoch"], "native": {"epoch": identity[0], "local_id": identity[1]},
                "continued": bool(row.get("identity_links"))}

    def _fresh(self, identity):
        if identity in self.active or identity in self.handed_off or identity in self.outside_seen:return False
        return self.archived is None or self.archived(vehicle_key(self.run_id, *identity)) is None

    def _continue_fragments(self, observations, timestamp, unassigned=None):
        """Link a never-seen native ID to a just-lost row when nothing else competes.

        Work is O(newborns x lost) gate arithmetic (motion fitted once per lost
        row) plus O(observations) per surviving pair, and capped per frame."""
        newborns = []
        for o in observations:
            if o.local_id is None or not o.stable_id or not o.in_region:continue
            identity = (o.epoch, o.local_id)
            height = _height(o)
            if height is not None and self._fresh(identity):newborns.append((identity, o, height))
        if not newborns:return
        present = {(o.epoch, o.local_id) for o in observations if o.local_id is not None and o.stable_id}
        lost = []
        # Lost stationary or short-lived vehicles within OCCLUSION_MAX_GAP_SEC cannot
        # link past STATIC_CONTINUATION_MAX_GAP_SEC (or with <3 sightings), but still
        # veto extended occlusion-tail links when any occlusion candidate is present.
        static_vetoes = []
        occlusion_count = 0
        native_exit = self.policy.exit_persistence_sec > 0
        veto_window = min(self.policy.max_gap_sec, OCCLUSION_MAX_GAP_SEC)+1e-9
        for identity, state in self.active.items():
            row, previous, history = state["row"], state["previous"], state.get("history")
            if identity in present or row["completed"] or previous is None:continue
            if row["vehicle_id"] not in self.records:continue
            if native_exit and state.get("exit_candidate") is not None:
                last_box = state.get("last_valid_box")
                if last_box is not None and 0 < timestamp-last_box[0] <= veto_window:
                    _, bx, by, bh = last_box
                    static_vetoes.append((identity, bx, by, by-bh, bh))
                continue
            if not history:continue
            dt = timestamp-previous["time"]
            if not 0 < dt <= self.policy.max_gap_sec+1e-9:continue
            last_time, x, y, height = history[-1]
            if last_time != previous["time"]:continue
            y1_last = y-height
            # A fragment seen in fewer than MIN_FIT_SAMPLES frames or a link-capped row
            # cannot continue onto a newborn inside the ROI, but still vetoes occlusion-tail links.
            if len(history) < MIN_FIT_SAMPLES or len(row.get("identity_links", ())) >= MAX_LINKS_PER_ROW:
                if dt <= OCCLUSION_MAX_GAP_SEC+1e-9:
                    static_vetoes.append((identity, x, y, y1_last, height))
                continue
            fitted_motion = _fit(history)
            # Require two consecutive earlier samples to support the taller baseline
            # (isolated tall outlier frames must not turn a stationary track into an occlusion tail).
            earlier = list(history)[:-1]
            baseline = max((a if a[3] <= b[3] else b for a, b in zip(earlier, earlier[1:])),
                           key=lambda sample: sample[3])
            _, _, y2_base, h_base = baseline
            occlusion_tail = False
            pred_vx = pred_vy = 0.
            fitted = False
            if fitted_motion is not None:
                vx, vy, vy_top = fitted_motion
                top_speed = math.hypot(vx, vy_top)
                dh_dt = vy-vy_top
                y1_vals = [s[2]-s[3] for s in history]
                x_vals = [s[1] for s in history]
                occlusion_tail = (
                    height <= OCCLUSION_SHRINK_RATIO*h_base
                    and dh_dt < -OCCLUSION_STATIC_TOP_SPEED*h_base
                    and (y2_base-y) >= OCCLUSION_BOTTOM_SHARE*(h_base-height)
                    and max(y1_vals)-min(y1_vals) <= OCCLUSION_TOP_DRIFT_HEIGHTS*h_base
                    and max(x_vals)-min(x_vals) <= OCCLUSION_TOP_DRIFT_HEIGHTS*h_base
                    and top_speed < OCCLUSION_STATIC_TOP_SPEED*h_base
                )
                # Whenever the bottom retreats upward while shrinking, qualify motion from
                # the top-edge fit against the taller baseline so shrinkage never promotes a
                # slow/stationary car to moving; when shrinkage dominates (vy_top > dh_dt),
                # predict vertical motion from the top edge too (preserving horizontal vx).
                shrinking_up = vy < 0 and dh_dt < 0
                bottom_clipped = shrinking_up and vy_top > dh_dt
                pred_vx = vx
                pred_vy = min(0., vy_top) if bottom_clipped else vy
                qual_vy = min(0., vy_top) if shrinking_up else vy
                speed_ref = max(s[3] for s in history) if shrinking_up else height
                fitted = not occlusion_tail and math.hypot(pred_vx, qual_vy) >= MIN_MOVING_HEIGHTS_PER_SEC*speed_ref
            if fitted:
                gate = CONTINUATION_BASE_HEIGHTS+CONTINUATION_HEIGHTS_PER_SEC*dt
                lost.append((identity, state, dt, (x+pred_vx*dt, y+pred_vy*dt), height, gate, True, None))
            elif occlusion_tail and dt <= OCCLUSION_MAX_GAP_SEC+1e-9:
                occlusion = (x, y1_last, y, height, y2_base, h_base)
                lost.append((identity, state, dt, (x, y1_last), h_base, OCCLUSION_TOP_GATE_HEIGHTS, False, occlusion))
                occlusion_count += 1
            elif dt <= STATIC_CONTINUATION_MAX_GAP_SEC+1e-9:
                lost.append((identity, state, dt, (x, y), height, CONTINUATION_BASE_HEIGHTS, False, None))
                static_vetoes.append((identity, x, y, y1_last, height))
            elif dt <= OCCLUSION_MAX_GAP_SEC+1e-9:
                static_vetoes.append((identity, x, y, y1_last, height))
            else:continue
        if not lost:return
        if unassigned is None:
            # Without the frame's withheld detections a competitor may be invisible: do not guess.
            self.continuation_unavailable_frames += 1
            return
        pair_checks = len(newborns)*len(lost)+min(len(newborns), occlusion_count)*len(static_vetoes)
        if pair_checks > MAX_PAIR_CHECKS:
            self.continuation_skipped_frames += 1
            return
        # Validate every supplied observation/source/unassigned box before matching:
        # any unreadable competitor box makes the frame's evidence unavailable.
        competitors = _competitors(observations, unassigned)
        if competitors is None:
            self.continuation_unavailable_frames += 1
            return
        # The raw detection a newborn was emitted from (``source_bbox``) is the same
        # vehicle seen before smoothing: it widens ambiguity, never acceptance.
        source_anchors = []
        for _, o, _ in newborns:
            raw = getattr(o, "source_bbox", None)
            source = None if raw is None else _raw_box(raw)
            if source is None:
                source_anchors.append(None)
            else:
                sx = (source[0]+source[2])/2
                source_anchors.append(((sx, source[3]), (sx, source[1])))
        # Degrees on position alone: class and size do not make a nearby box less of a competitor.
        new_degree = [0]*len(newborns); lost_degree = [0]*len(lost); edge = {}
        for i, (new, o, new_height) in enumerate(newborns):
            px, py = float(o.point[0]), float(o.point[1])
            top_point = (px, py-new_height)
            anchors = source_anchors[i]
            for j, (old, _, _, predicted, scale_h, gate, _, occlusion) in enumerate(lost):
                if old[0] != new[0]:continue
                if occlusion is None:
                    error = math.hypot(predicted[0]-px, predicted[1]-py)/scale_h
                    in_emitted = error <= gate
                    in_source = anchors is not None and math.hypot(
                        predicted[0]-anchors[0][0], predicted[1]-anchors[0][1])/scale_h <= gate
                else:
                    error = math.hypot(predicted[0]-top_point[0], predicted[1]-top_point[1])/scale_h
                    in_emitted = error <= gate
                    in_source = anchors is not None and math.hypot(
                        predicted[0]-anchors[1][0], predicted[1]-anchors[1][1])/scale_h <= gate
                if in_emitted or in_source:
                    new_degree[i] += 1; lost_degree[j] += 1
                    if in_emitted:edge[i] = (j, error)
        for i, (j, error) in edge.items():
            if new_degree[i] != 1 or lost_degree[j] != 1:continue
            new, o, new_height = newborns[i]
            old, state, dt, predicted, scale_h, gate, fitted, occlusion = lost[j]
            if state["row"]["class_name"] != o.class_name:continue
            if occlusion is None:
                if max(scale_h, new_height)/min(scale_h, new_height) > MAX_HEIGHT_RATIO:continue
                if self._competed(o, new_height, predicted, scale_h, gate, competitors):continue
                self._link(old, new, state, o, timestamp, dt, error, gate, predicted, fitted)
            else:
                ox, oy1, oy2_last, h_last, oy2_base, h_base = occlusion
                if new_height < h_last/MAX_HEIGHT_RATIO or new_height > h_base*OCCLUSION_MAX_RECOVERY_RATIO:continue
                if not (oy2_last-OCCLUSION_BOTTOM_MARGIN_HEIGHTS*h_base <= o.point[1] <= oy2_base+OCCLUSION_BOTTOM_MARGIN_HEIGHTS*h_base):continue
                if self._vetoed_by_lost_static(new, o, new_height, source_anchors[i], static_vetoes):continue
                if self._competed_occlusion(o, new_height, predicted, occlusion, gate, competitors):continue
                observed = (float(o.point[0]), float(o.point[1])-new_height)
                self._link(old, new, state, o, timestamp, dt, error, gate, predicted, fitted,
                           observed=observed, rule="mutually_unique_uncontested_occlusion_tail_gate")

    @staticmethod
    def _vetoed_by_lost_static(new, newborn, new_height, source_anchors, static_vetoes):
        """A nearby lost stationary vehicle within the extended occlusion window makes
        an extended occlusion link ambiguous even after its own 0.10 s link cap elapsed."""
        px, py = float(newborn.point[0]), float(newborn.point[1])
        top = (px, py-new_height)
        for other_id, sx, sy, sy1, sh in static_vetoes:
            if other_id[0] != new[0]:continue
            if math.hypot(sx-px, sy-py)/sh <= CONTINUATION_BASE_HEIGHTS:return True
            if math.hypot(sx-top[0], sy1-top[1])/sh <= OCCLUSION_TOP_GATE_HEIGHTS:return True
            if source_anchors is not None:
                s_fp, s_tp = source_anchors
                if math.hypot(sx-s_fp[0], sy-s_fp[1])/sh <= CONTINUATION_BASE_HEIGHTS:return True
                if math.hypot(sx-s_tp[0], sy1-s_tp[1])/sh <= OCCLUSION_TOP_GATE_HEIGHTS:return True
        return False

    @staticmethod
    def _newborn_boxes(newborn, new_height):
        boxes = []
        box = None if newborn.bbox is None else _raw_box(newborn.bbox)
        if box is not None:
            px, py = float(newborn.point[0]), float(newborn.point[1])
            boxes.append(((px, py), (px, py-new_height), new_height, box))
        source = getattr(newborn, "source_bbox", None)
        source = None if source is None else _raw_box(source)
        if source is not None and source != box:
            sx = (source[0]+source[2])/2
            boxes.append(((sx, source[3]), (sx, source[1]), source[3]-source[1], source))
        return boxes

    @classmethod
    def _competed(cls, newborn, new_height, predicted, height, gate, competitors):
        # The newborn's emitted box and its raw source box each compete with every
        # other box (and its raw source): a smoothed box can drift off a neighbour
        # its detection overlaps.
        newborn_boxes = cls._newborn_boxes(newborn, new_height)
        for other, point, other_height, bbox in competitors:
            if other is newborn:continue
            if math.dist(predicted, point)/height <= gate:return True
            if not newborn_boxes or bbox is None or other_height is None:continue
            for own_point, _, own_height, own_box in newborn_boxes:
                if (_overlap_of_smaller(own_box, bbox) >= COMPETITOR_OVERLAP and
                        math.dist(own_point, point)/min(own_height, other_height) <= COMPETITOR_FOOTPOINT_HEIGHTS):
                    return True
        return False

    @classmethod
    def _competed_occlusion(cls, newborn, new_height, predicted, occlusion, gate, competitors):
        ox, oy1, oy2_last, h_last, oy2_base, h_base = occlusion
        newborn_boxes = cls._newborn_boxes(newborn, new_height)
        for other, point, other_height, bbox in competitors:
            if other is newborn:continue
            other_top = (point[0], point[1]-other_height) if other_height is not None else point
            if math.dist(predicted, other_top)/h_base <= gate:return True
            if math.dist((ox, oy2_last), point)/h_last <= CONTINUATION_BASE_HEIGHTS:return True
            if math.dist((ox, oy2_base), point)/h_base <= gate:return True
            if not newborn_boxes or bbox is None or other_height is None:continue
            for own_fp, own_tp, own_height, own_box in newborn_boxes:
                if _overlap_of_smaller(own_box, bbox) >= COMPETITOR_OVERLAP:
                    smaller = min(own_height, other_height)
                    if (math.dist(own_fp, point)/smaller <= COMPETITOR_FOOTPOINT_HEIGHTS or
                            math.dist(own_tp, other_top)/smaller <= COMPETITOR_FOOTPOINT_HEIGHTS):
                        return True
        return False

    def _link(self, old, new, state, observation, timestamp, dt, error, gate, predicted, fitted, *,
              observed=None, rule="mutually_unique_uncontested_motion_gate"):
        row = state["row"]
        own = (row["tracker_epoch"], row["local_id"])
        del self.active[old]
        self.handed_off[old] = {"row_id": row["vehicle_id"], "segments": state["segments"] if old == own else 0}
        self.active[new] = state
        # The handoff is a heuristic: nothing (wait, crossing, motion) is carried across it.
        state["previous"] = dict(state["previous"], handoff=True)
        state["history"].clear()
        state["stopped"] = False; state["stop_candidate"] = state["move_candidate"] = None
        self._clear_clip_state(state)
        state["last_valid_box"] = None
        self._clear_native_exit(state)
        native = {"epoch": new[0], "local_id": new[1]}
        # Added on the first link only, so rows of unchanged runs keep their schema.
        row.setdefault("native_ids", [{"epoch": own[0], "local_id": own[1]}]).append(native)
        row.setdefault("identity_links", []).append({"from": {"epoch": old[0], "local_id": old[1]}, "to": native,
            "lost_at": state["previous"]["time"], "linked_at": timestamp, "gap_sec": dt,
            "predicted_point": list(predicted), "observed_point": list(observation.point if observed is None else observed),
            "error_heights": error, "gate_heights": gate, "moving_fit": fitted,
            "rule": rule})
        self._flag(row, "identity_continued")

    def invalidate(self, timestamp, reason):
        for state in self.active.values():
            row = state["row"]
            if not row["completed"]:
                self._flag(row, reason)
                state["previous"] = None
                state["stop_candidate"] = state["move_candidate"] = None
                self._clear_clip_state(state)
                state["last_valid_box"] = None
                self._clear_native_exit(state)
                if "history" in state:state["history"].clear()
                row["tracking_state"] = "UNKNOWN"

    def close(self, timestamp, reason="recording_end"):
        self.invalidate(timestamp, reason)
        for row in self.records.values():
            if not row["completed"]: row.update(completed=True, completed_at=timestamp);self.work+=1

    def rows(self, offset=0, limit=100, include_intervals=False):
        rows = list(self.records.values())
        return [{k: (round(v, 6) if isinstance(v, float) else deepcopy(v))
                 for k, v in row.items() if include_intervals or k != "intervals"}
                for row in rows[offset:offset+limit]]
