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


_MAX_PIXEL_COORD = 1e15


def _finite_point(point):
    try:px, py = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError, OverflowError, KeyError):return False
    return math.isfinite(px) and math.isfinite(py) and abs(px) <= _MAX_PIXEL_COORD and abs(py) <= _MAX_PIXEL_COORD


def _height(observation):
    height = observation.height
    if height is None and observation.bbox is not None:
        box = _raw_box(observation.bbox)
        height = None if box is None else box[3]-box[1]
    try:height = float(height)
    except (TypeError, ValueError, OverflowError):return None
    return height if math.isfinite(height) and 0 < height <= 2.0*_MAX_PIXEL_COORD else None


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
    return ((x1, y1, x2, y2) if all(map(math.isfinite, (x1, y1, x2, y2, w, h, (x1+x2)/2, w*h))) and
            w > 0 and h > 0 and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= _MAX_PIXEL_COORD else None)


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
        except (TypeError, ValueError, IndexError, OverflowError, KeyError):return None
        if not (math.isfinite(px) and math.isfinite(py) and abs(px) <= _MAX_PIXEL_COORD and abs(py) <= _MAX_PIXEL_COORD):return None
        h = _height(o)
        if o.height is not None and h is None:return None
        if h is not None and not (math.isfinite(py-h) and abs(py-h) <= _MAX_PIXEL_COORD):return None
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


_DD_SPLIT = 134217729.0
_BSHARE_HI = (lambda t: t-(t-OCCLUSION_BOTTOM_SHARE))(_DD_SPLIT*OCCLUSION_BOTTOM_SHARE)
_BSHARE_LO = OCCLUSION_BOTTOM_SHARE-_BSHARE_HI
# Power-of-two scale (2**200) keeps Dekker TwoProd and greedy math.fsum expansions
# strictly normal with zero overflow (<= 2**251 << 2**1024) and zero subnormal
# underflow (>= 2**-979 >> 2**-1022) across the entire float64 domain down to 5e-324.
_REAR_SCALE = 200


def _exact_rear(y2, h):
    """Exact 3-term expansion ``(c0, c1, c2)`` of ``2**200 * (y2 - OCCLUSION_BOTTOM_SHARE * h)``
    with zero residual for all finite ``|y2| <= _MAX_PIXEL_COORD``, ``0 < h <= 2*_MAX_PIXEL_COORD``."""
    ys = math.ldexp(y2, _REAR_SCALE)
    hs = math.ldexp(h, _REAR_SCALE)
    p_hi = OCCLUSION_BOTTOM_SHARE*hs
    ht = _DD_SPLIT*hs
    h_hi = ht-(ht-hs)
    h_lo = hs-h_hi
    p_lo = ((_BSHARE_HI*h_hi-p_hi)+_BSHARE_HI*h_lo+_BSHARE_LO*h_hi)+_BSHARE_LO*h_lo
    c0 = math.fsum((ys, -p_hi, -p_lo))
    c1 = math.fsum((ys, -p_hi, -p_lo, -c0))
    c2 = math.fsum((ys, -p_hi, -p_lo, -c0, -c1))
    return c0, c1, c2


def _exact_floor(py, h, eps):
    """Exact 4-term expansion ``(f0, f1, f2, f3)`` of ``2**200 * (py - OCCLUSION_BOTTOM_SHARE * h - eps)``
    so ``(c0, c1, c2, 0.0) >= (f0, f1, f2, f3)`` matches infinite-precision arithmetic everywhere."""
    ys = math.ldexp(py, _REAR_SCALE)
    hs = math.ldexp(h, _REAR_SCALE)
    es = math.ldexp(eps, _REAR_SCALE)
    p_hi = OCCLUSION_BOTTOM_SHARE*hs
    ht = _DD_SPLIT*hs
    h_hi = ht-(ht-hs)
    h_lo = hs-h_hi
    p_lo = ((_BSHARE_HI*h_hi-p_hi)+_BSHARE_HI*h_lo+_BSHARE_LO*h_hi)+_BSHARE_LO*h_lo
    f0 = math.fsum((ys, -p_hi, -p_lo, -es))
    f1 = math.fsum((ys, -p_hi, -p_lo, -es, -f0))
    f2 = math.fsum((ys, -p_hi, -p_lo, -es, -f0, -f1))
    f3 = math.fsum((ys, -p_hi, -p_lo, -es, -f0, -f1, -f2))
    return f0, f1, f2, f3


def _rear_err(y2, h):
    """Conservative float64 rearrangement slack between ``(y2-py)-0.7*(h-height)``
    and ``(y2-0.7*h)-(py-0.7*height)`` so dominance never drops cutoff candidates."""
    return max(1e-12, 16.0*math.ulp(max(abs(y2), abs(h), 1.0)))


def _dominates_coords(ax, ay1, ay2, ah, arear, aerr, bx, by1, by2, bh, brear, berr,
                      arlo=0.0, brlo=0.0, arlo2=0.0, brlo2=0.0):
    """True when stationary baseline ``a`` provably covers ``b`` in height, rear
    bottom-share score (exact 3-term expansion or with rearrangement slack), and (x, y1) drift bounds."""
    if ax == bx and ay1 == by1 and ay2 == by2 and ah == bh:
        return True
    if ah < bh:
        return False
    if aerr == 0.0 and berr == 0.0:
        if arear < brear or (arear == brear and (arlo < brlo or (arlo == brlo and arlo2 < brlo2))):
            return False
    elif arear-aerr < brear+berr:
        return False
    ra = OCCLUSION_TOP_DRIFT_HEIGHTS*ah
    rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
    return (ax-ra <= bx-rb and ax+ra >= bx+rb and
            ay1-ra <= by1-rb and ay1+ra >= by1+rb)


_R_LEAF_MAX = 8
_R_NODE_MAX = 16


class _RNode:
    """Self-balancing 2D R-tree node with height/rear/drift envelopes and strict
    O(log N) split-only insertion (zero whole-subtree rebuilds)."""
    __slots__ = ("children", "idxs", "count", "min_slot",
                 "min_x", "max_x", "min_y", "max_y", "min_h", "max_h",
                 "max_rear_lo", "min_rear_hi",
                 "min_x_hi", "max_x_hi", "min_x_lo", "max_x_lo",
                 "min_y_hi", "max_y_hi", "min_y_lo", "max_y_lo")

    def __init__(self, *, idxs=None, children=None):
        self.idxs = idxs
        self.children = children
        self.count = 0
        self.min_slot = math.inf
        self.min_x = self.min_y = self.min_h = math.inf
        self.max_x = self.max_y = self.max_h = -math.inf
        self.max_rear_lo = -math.inf; self.min_rear_hi = math.inf
        self.min_x_hi = self.min_x_lo = self.min_y_hi = self.min_y_lo = math.inf
        self.max_x_hi = self.max_x_lo = self.max_y_hi = self.max_y_lo = -math.inf

    def include_point(self, slot, bx, by1, bh, brear, berr):
        if slot < self.min_slot: self.min_slot = slot
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
            self.min_slot = math.inf
            self.min_x = self.min_y = self.min_h = math.inf
            self.max_x = self.max_y = self.max_h = -math.inf
            self.max_rear_lo = -math.inf; self.min_rear_hi = math.inf
            self.min_x_hi = self.min_x_lo = self.min_y_hi = self.min_y_lo = math.inf
            self.max_x_hi = self.max_x_lo = self.max_y_hi = self.max_y_lo = -math.inf
            return
        min_slot = math.inf
        min_x = min_y = min_h = math.inf
        max_x = max_y = max_h = -math.inf
        max_rear_lo = -math.inf; min_rear_hi = math.inf
        min_x_hi = min_x_lo = min_y_hi = min_y_lo = math.inf
        max_x_hi = max_x_lo = max_y_hi = max_y_lo = -math.inf
        for i in idxs:
            if i < min_slot: min_slot = i
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
        self.min_slot = min_slot
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
        ch = self.children
        c0 = ch[0]
        min_slot = c0.min_slot
        min_x = c0.min_x; max_x = c0.max_x
        min_y = c0.min_y; max_y = c0.max_y
        min_h = c0.min_h; max_h = c0.max_h
        max_rear_lo = c0.max_rear_lo; min_rear_hi = c0.min_rear_hi
        min_x_hi = c0.min_x_hi; max_x_hi = c0.max_x_hi
        min_x_lo = c0.min_x_lo; max_x_lo = c0.max_x_lo
        min_y_hi = c0.min_y_hi; max_y_hi = c0.max_y_hi
        min_y_lo = c0.min_y_lo; max_y_lo = c0.max_y_lo
        count = c0.count
        for c in ch[1:]:
            if c.min_slot < min_slot: min_slot = c.min_slot
            if c.min_x < min_x: min_x = c.min_x
            if c.max_x > max_x: max_x = c.max_x
            if c.min_y < min_y: min_y = c.min_y
            if c.max_y > max_y: max_y = c.max_y
            if c.min_h < min_h: min_h = c.min_h
            if c.max_h > max_h: max_h = c.max_h
            if c.max_rear_lo > max_rear_lo: max_rear_lo = c.max_rear_lo
            if c.min_rear_hi < min_rear_hi: min_rear_hi = c.min_rear_hi
            if c.min_x_hi < min_x_hi: min_x_hi = c.min_x_hi
            if c.max_x_hi > max_x_hi: max_x_hi = c.max_x_hi
            if c.min_x_lo < min_x_lo: min_x_lo = c.min_x_lo
            if c.max_x_lo > max_x_lo: max_x_lo = c.max_x_lo
            if c.min_y_hi < min_y_hi: min_y_hi = c.min_y_hi
            if c.max_y_hi > max_y_hi: max_y_hi = c.max_y_hi
            if c.min_y_lo < min_y_lo: min_y_lo = c.min_y_lo
            if c.max_y_lo > max_y_lo: max_y_lo = c.max_y_lo
            count += c.count
        self.min_slot = min_slot
        self.min_x = min_x; self.max_x = max_x
        self.min_y = min_y; self.max_y = max_y
        self.min_h = min_h; self.max_h = max_h
        self.max_rear_lo = max_rear_lo; self.min_rear_hi = min_rear_hi
        self.min_x_hi = min_x_hi; self.max_x_hi = max_x_hi
        self.min_x_lo = min_x_lo; self.max_x_lo = max_x_lo
        self.min_y_hi = min_y_hi; self.max_y_hi = max_y_hi
        self.min_y_lo = min_y_lo; self.max_y_lo = max_y_lo
        self.count = count


_BBLOCK_MAX = 512
_HBLOCK_MAX = 64
_HNODE_MAX = 16
_HPOS_DEAD = 0xFFFF


_FIFO_BIAS = 1 << 31


class _SlotFIFO:
    """Fixed-depth radix-tree FIFO (4x64 internal levels + 256-int C-array leaves,
    4.29B slot span) with strict worst-case O(1) append/appendleft/pop/popleft and
    O(1) indexed reads (zero directory resizing)."""
    __slots__ = ("_root", "_lo", "_hi", "base")

    def __init__(self):
        self._root = None
        self._lo = 0
        self._hi = 0
        self.base = 0

    def __len__(self):
        return self._hi-self._lo

    def __bool__(self):
        return self._hi > self._lo

    def __getitem__(self, idx):
        u = self._lo+idx+_FIFO_BIAS
        return self._root[(u >> 26) & 63][(u >> 20) & 63][(u >> 14) & 63][(u >> 8) & 63][u & 255]

    def __iter__(self):
        root = self._root
        pos = self._lo; end = self._hi
        while pos < end:
            u = pos+_FIFO_BIAS
            chunk = root[(u >> 26) & 63][(u >> 20) & 63][(u >> 14) & 63][(u >> 8) & 63]
            lo = u & 255
            hi = min(256, lo+(end-pos))
            for i in range(lo, hi):
                yield chunk[i]
            pos += hi-lo

    def clear(self):
        self._root = None
        self._lo = self._hi = self.base = 0

    def retire_to(self, sink):
        if self._root is not None:
            if self._hi-self._lo > 512:
                sink.append((0, self._root))
            self._root = None
        self._lo = self._hi = self.base = 0

    def _set_pos(self, pos, val):
        u = pos+_FIFO_BIAS
        root = self._root
        if root is None:
            root = [None]*64
            self._root = root
        i0 = (u >> 26) & 63
        n1 = root[i0]
        if n1 is None:
            n1 = [None]*64
            root[i0] = n1
        i1 = (u >> 20) & 63
        n2 = n1[i1]
        if n2 is None:
            n2 = [None]*64
            n1[i1] = n2
        i2 = (u >> 14) & 63
        n3 = n2[i2]
        if n3 is None:
            n3 = [None]*64
            n2[i2] = n3
        i3 = (u >> 8) & 63
        leaf = n3[i3]
        if leaf is None:
            leaf = array("i", (0,))*256
            n3[i3] = leaf
        leaf[u & 255] = val

    def append(self, val):
        pos = self._hi
        self._hi = pos+1
        self._set_pos(pos, val)

    def appendleft(self, val):
        pos = self._lo-1
        self._lo = pos
        self._set_pos(pos, val)

    def _drop_empty(self, pos):
        u = pos+_FIFO_BIAS
        u_lo = self._lo+_FIFO_BIAS
        u_hi = self._hi-1+_FIFO_BIAS
        root = self._root
        i0 = (u >> 26) & 63; n1 = root[i0]
        i1 = (u >> 20) & 63; n2 = n1[i1]
        i2 = (u >> 14) & 63; n3 = n2[i2]
        n3[(u >> 8) & 63] = None
        if (u_lo >> 14) > (u >> 14) or (u_hi >> 14) < (u >> 14):
            n2[i2] = None
            if (u_lo >> 20) > (u >> 20) or (u_hi >> 20) < (u >> 20):
                n1[i1] = None
                if (u_lo >> 26) > (u >> 26) or (u_hi >> 26) < (u >> 26):
                    root[i0] = None

    def pop(self):
        pos = self._hi-1
        self._hi = pos
        u = pos+_FIFO_BIAS
        val = self._root[(u >> 26) & 63][(u >> 20) & 63][(u >> 14) & 63][(u >> 8) & 63][u & 255]
        if self._lo == self._hi:
            self._root = None
            self._lo = self._hi = 0
        elif (u & 255) == 0:
            self._drop_empty(pos)
        return val

    def popleft(self):
        pos = self._lo
        self._lo = pos+1
        self.base += 1
        u = pos+_FIFO_BIAS
        val = self._root[(u >> 26) & 63][(u >> 20) & 63][(u >> 14) & 63][(u >> 8) & 63][u & 255]
        if self._lo == self._hi:
            self._root = None
            self._lo = self._hi = 0
        elif (u & 255) == 255:
            self._drop_empty(pos)
        return val


class _BoundBlocks:
    """2-level sorted C-array blocks of ``(bound, slot)`` for O(1) stationary checks,
    O(log B) inserts, and O(k) bulk-prefix drift pruning without touching surviving entries."""
    __slots__ = ("blocks", "max_vals")

    def __init__(self):
        self.blocks = []
        self.max_vals = array("d")

    def clear(self):
        self.blocks.clear()
        self.max_vals = array("d")

    def push(self, val, slot):
        blocks = self.blocks; max_vals = self.max_vals
        if not blocks:
            blocks.append((array("d", (val,)), array("I", (slot,))))
            max_vals.append(val)
            return
        b_idx = bisect.bisect_left(max_vals, val)
        if b_idx == len(blocks):
            b_idx -= 1
        vals, slots = blocks[b_idx]
        i = bisect.bisect_left(vals, val)
        vals.insert(i, val); slots.insert(i, slot)
        if i == len(vals)-1:
            max_vals[b_idx] = val
        if len(vals) > _BBLOCK_MAX:
            mid = len(vals)//2
            right = (vals[mid:], slots[mid:])
            del vals[mid:]; del slots[mid:]
            max_vals[b_idx] = vals[-1]
            blocks.insert(b_idx+1, right)
            max_vals.insert(b_idx+1, right[0][-1])

    def pop_below(self, threshold, alive, blk_by_id, slot_bid, dom_pending, cover_pending,
                  rears, errs, rlos, rlo2s, hpos, dirty_blks):
        blocks = self.blocks; max_vals = self.max_vals
        if not blocks or blocks[0][0][0] >= threshold:
            return 0, 0
        killed = 0; cov_killed = 0
        cut_b = bisect.bisect_left(max_vals, threshold)
        if cut_b:
            for b_i in range(cut_b):
                for s in blocks[b_i][1]:
                    if alive[s]:
                        alive[s] = 0
                        blk_by_id[slot_bid[s]].heap_remove(s, rears, errs, rlos, rlo2s, hpos, dirty=dirty_blks)
                        dom_pending[s] = 0
                        if cover_pending[s]:
                            cover_pending[s] = 0
                            cov_killed += 1
                        killed += 1
            del blocks[:cut_b]; del max_vals[:cut_b]
        if blocks and blocks[0][0][0] < threshold:
            vals, slots = blocks[0]
            cut = bisect.bisect_left(vals, threshold)
            for j in range(cut):
                s = slots[j]
                if alive[s]:
                    alive[s] = 0
                    blk_by_id[slot_bid[s]].heap_remove(s, rears, errs, rlos, rlo2s, hpos, dirty=dirty_blks)
                    dom_pending[s] = 0
                    if cover_pending[s]:
                        cover_pending[s] = 0
                        cov_killed += 1
                    killed += 1
            if cut == len(vals):
                del blocks[0]; del max_vals[0]
            else:
                del vals[:cut]; del slots[:cut]
        return killed, cov_killed


def _h_propagate_rear_up(node, hi, lo, lo2):
    while node is not None and (hi > node.max_rear_hi or
                                (hi == node.max_rear_hi and (lo > node.max_rear_lo or
                                                             (lo == node.max_rear_lo and lo2 > node.max_rear_lo2)))):
        node.max_rear_hi = hi; node.max_rear_lo = lo; node.max_rear_lo2 = lo2
        node = node.parent


def _h_propagate_rear_down(node, old_hi, old_lo, old_lo2):
    while (node is not None and node.max_rear_hi == old_hi and
           node.max_rear_lo == old_lo and node.max_rear_lo2 == old_lo2):
        m_hi = -math.inf; m_lo = 0.0; m_lo2 = 0.0
        for c in node.children:
            chi = c.max_rear_hi; clo = c.max_rear_lo; clo2 = c.max_rear_lo2
            if chi > m_hi or (chi == m_hi and (clo > m_lo or (clo == m_lo and clo2 > m_lo2))):
                m_hi = chi; m_lo = clo; m_lo2 = clo2
        if m_hi == old_hi and m_lo == old_lo and m_lo2 == old_lo2:
            return
        node.max_rear_hi = m_hi; node.max_rear_lo = m_lo; node.max_rear_lo2 = m_lo2
        node = node.parent


def _h_propagate_bounds_up(node, x_hi, neg_x_lo, y_hi, neg_y_lo):
    while node is not None:
        changed = False
        if x_hi > node.max_x_hi: node.max_x_hi = x_hi; changed = True
        if neg_x_lo > node.max_neg_x_lo: node.max_neg_x_lo = neg_x_lo; changed = True
        if y_hi > node.max_y_hi: node.max_y_hi = y_hi; changed = True
        if neg_y_lo > node.max_neg_y_lo: node.max_neg_y_lo = neg_y_lo; changed = True
        if not changed:
            return
        node = node.parent


def _h_propagate_bounds_down(node, old_x_hi, old_neg_x_lo, old_y_hi, old_neg_y_lo):
    while node is not None:
        if not (node.max_x_hi == old_x_hi or node.max_neg_x_lo == old_neg_x_lo or
                node.max_y_hi == old_y_hi or node.max_neg_y_lo == old_neg_y_lo):
            return
        m_xh = m_nxl = m_yh = m_nyl = -math.inf
        for c in node.children:
            if c.max_x_hi > m_xh: m_xh = c.max_x_hi
            if c.max_neg_x_lo > m_nxl: m_nxl = c.max_neg_x_lo
            if c.max_y_hi > m_yh: m_yh = c.max_y_hi
            if c.max_neg_y_lo > m_nyl: m_nyl = c.max_neg_y_lo
        if (m_xh == node.max_x_hi and m_nxl == node.max_neg_x_lo and
                m_yh == node.max_y_hi and m_nyl == node.max_neg_y_lo):
            return
        old_x_hi = node.max_x_hi if m_xh < node.max_x_hi else math.inf
        old_neg_x_lo = node.max_neg_x_lo if m_nxl < node.max_neg_x_lo else math.inf
        old_y_hi = node.max_y_hi if m_yh < node.max_y_hi else math.inf
        old_neg_y_lo = node.max_neg_y_lo if m_nyl < node.max_neg_y_lo else math.inf
        node.max_x_hi = m_xh; node.max_neg_x_lo = m_nxl
        node.max_y_hi = m_yh; node.max_neg_y_lo = m_nyl
        node = node.parent


def _h_refresh_ancestors(nodes):
    """Refresh live rear maxima and 4-axis drift envelopes bottom-up once per touched _HNode."""
    cur_level = nodes
    while cur_level:
        next_level = set()
        for node in cur_level:
            m_hi = -math.inf; m_lo = 0.0; m_lo2 = 0.0
            m_xh = m_nxl = m_yh = m_nyl = -math.inf
            for c in node.children:
                chi = c.max_rear_hi; clo = c.max_rear_lo; clo2 = c.max_rear_lo2
                if chi > m_hi or (chi == m_hi and (clo > m_lo or (clo == m_lo and clo2 > m_lo2))):
                    m_hi = chi; m_lo = clo; m_lo2 = clo2
                if c.max_x_hi > m_xh: m_xh = c.max_x_hi
                if c.max_neg_x_lo > m_nxl: m_nxl = c.max_neg_x_lo
                if c.max_y_hi > m_yh: m_yh = c.max_y_hi
                if c.max_neg_y_lo > m_nyl: m_nyl = c.max_neg_y_lo
            if (m_hi != node.max_rear_hi or m_lo != node.max_rear_lo or m_lo2 != node.max_rear_lo2 or
                    m_xh != node.max_x_hi or m_nxl != node.max_neg_x_lo or
                    m_yh != node.max_y_hi or m_nyl != node.max_neg_y_lo):
                node.max_rear_hi = m_hi; node.max_rear_lo = m_lo; node.max_rear_lo2 = m_lo2
                node.max_x_hi = m_xh; node.max_neg_x_lo = m_nxl
                node.max_y_hi = m_yh; node.max_neg_y_lo = m_nyl
                if node.parent is not None:
                    next_level.add(node.parent)
        cur_level = next_level


class _HBlock:
    __slots__ = ("h", "y2", "x", "slots", "bid",
                 "rear_heap", "max_rear_hi", "max_rear_lo", "max_rear_lo2", "live_count", "parent", "next_blk",
                 "max_x_hi", "max_x_hi_slot", "max_neg_x_lo", "max_neg_x_lo_slot",
                 "max_y_hi", "max_y_hi_slot", "max_neg_y_lo", "max_neg_y_lo_slot", "bounds_dirty")

    def __init__(self, h=None, y2=None, x=None, slots=None,
                 rear_heap=None, max_rear_hi=-math.inf, max_rear_lo=0.0, max_rear_lo2=0.0,
                 live_count=0, parent=None, next_blk=None, bid=0):
        self.h = array("d") if h is None else h
        self.y2 = array("d") if y2 is None else y2
        self.x = array("d") if x is None else x
        self.slots = array("I") if slots is None else slots
        self.bid = bid
        self.rear_heap = array("I") if rear_heap is None else rear_heap
        self.max_rear_hi = max_rear_hi
        self.max_rear_lo = max_rear_lo
        self.max_rear_lo2 = max_rear_lo2
        self.live_count = live_count
        self.parent = parent
        self.next_blk = next_blk
        self.max_x_hi = self.max_neg_x_lo = self.max_y_hi = self.max_neg_y_lo = -math.inf
        self.max_x_hi_slot = self.max_neg_x_lo_slot = self.max_y_hi_slot = self.max_neg_y_lo_slot = -1
        self.bounds_dirty = 0

    def heap_push(self, slot, bx, by1, bh, rears, errs, rlos, rlo2s, hpos):
        heap = self.rear_heap
        pos = len(heap)
        heap.append(slot)
        v_hi = rears[slot]+errs[slot]; v_lo = rlos[slot]; v_lo2 = rlo2s[slot]
        while pos > 0:
            p = (pos-1) >> 1
            ps = heap[p]
            p_hi = rears[ps]+errs[ps]; p_lo = rlos[ps]; p_lo2 = rlo2s[ps]
            if p_hi > v_hi or (p_hi == v_hi and (p_lo > v_lo or (p_lo == v_lo and p_lo2 >= v_lo2))):
                break
            heap[pos] = ps
            hpos[ps] = pos
            pos = p
        heap[pos] = slot
        hpos[slot] = pos
        self.live_count = len(heap)
        if pos == 0:
            o_hi = self.max_rear_hi; o_lo = self.max_rear_lo; o_lo2 = self.max_rear_lo2
            self.max_rear_hi = v_hi; self.max_rear_lo = v_lo; self.max_rear_lo2 = v_lo2
            if v_hi > o_hi or (v_hi == o_hi and (v_lo > o_lo or (v_lo == o_lo and v_lo2 > o_lo2))):
                _h_propagate_rear_up(self.parent, v_hi, v_lo, v_lo2)
        rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        xh = bx+rb; nxl = -(bx-rb); yh = by1+rb; nyl = -(by1-rb)
        b_up = False
        if xh > self.max_x_hi: self.max_x_hi = xh; self.max_x_hi_slot = slot; b_up = True
        if nxl > self.max_neg_x_lo: self.max_neg_x_lo = nxl; self.max_neg_x_lo_slot = slot; b_up = True
        if yh > self.max_y_hi: self.max_y_hi = yh; self.max_y_hi_slot = slot; b_up = True
        if nyl > self.max_neg_y_lo: self.max_neg_y_lo = nyl; self.max_neg_y_lo_slot = slot; b_up = True
        if b_up:
            _h_propagate_bounds_up(self.parent, self.max_x_hi, self.max_neg_x_lo, self.max_y_hi, self.max_neg_y_lo)

    def refresh_bounds(self, xs, y1s, hs, propagate=True):
        raw_mask = self.bounds_dirty
        self.bounds_dirty = 0
        rear_dropped = bool(raw_mask & 16)
        mask = raw_mask & 15
        if not mask:
            return rear_dropped
        old_xh = self.max_x_hi; old_nxl = self.max_neg_x_lo
        old_yh = self.max_y_hi; old_nyl = self.max_neg_y_lo
        heap = self.rear_heap
        if not heap:
            self.max_x_hi = self.max_neg_x_lo = self.max_y_hi = self.max_neg_y_lo = -math.inf
            self.max_x_hi_slot = self.max_neg_x_lo_slot = self.max_y_hi_slot = self.max_neg_y_lo_slot = -1
        else:
            drift = OCCLUSION_TOP_DRIFT_HEIGHTS
            if mask == 1:
                m = -math.inf; sl = -1
                for s in heap:
                    v = xs[s]+drift*hs[s]
                    if v > m: m = v; sl = s
                self.max_x_hi = m; self.max_x_hi_slot = sl
            elif mask == 2:
                m = -math.inf; sl = -1
                for s in heap:
                    v = drift*hs[s]-xs[s]
                    if v > m: m = v; sl = s
                self.max_neg_x_lo = m; self.max_neg_x_lo_slot = sl
            elif mask == 4:
                m = -math.inf; sl = -1
                for s in heap:
                    v = y1s[s]+drift*hs[s]
                    if v > m: m = v; sl = s
                self.max_y_hi = m; self.max_y_hi_slot = sl
            elif mask == 8:
                m = -math.inf; sl = -1
                for s in heap:
                    v = drift*hs[s]-y1s[s]
                    if v > m: m = v; sl = s
                self.max_neg_y_lo = m; self.max_neg_y_lo_slot = sl
            else:
                m_xh = m_nxl = m_yh = m_nyl = -math.inf
                s_xh = s_nxl = s_yh = s_nyl = -1
                for s in heap:
                    bx = xs[s]; by1 = y1s[s]; rb = drift*hs[s]
                    xh = bx+rb; nxl = rb-bx; yh = by1+rb; nyl = rb-by1
                    if xh > m_xh: m_xh = xh; s_xh = s
                    if nxl > m_nxl: m_nxl = nxl; s_nxl = s
                    if yh > m_yh: m_yh = yh; s_yh = s
                    if nyl > m_nyl: m_nyl = nyl; s_nyl = s
                self.max_x_hi = m_xh; self.max_x_hi_slot = s_xh
                self.max_neg_x_lo = m_nxl; self.max_neg_x_lo_slot = s_nxl
                self.max_y_hi = m_yh; self.max_y_hi_slot = s_yh
                self.max_neg_y_lo = m_nyl; self.max_neg_y_lo_slot = s_nyl
        changed = (self.max_x_hi < old_xh or self.max_neg_x_lo < old_nxl or
                   self.max_y_hi < old_yh or self.max_neg_y_lo < old_nyl)
        if propagate and changed:
            _h_propagate_bounds_down(
                self.parent,
                old_xh if self.max_x_hi < old_xh else math.inf,
                old_nxl if self.max_neg_x_lo < old_nxl else math.inf,
                old_yh if self.max_y_hi < old_yh else math.inf,
                old_nyl if self.max_neg_y_lo < old_nyl else math.inf,
            )
        return changed or rear_dropped

    def heap_remove(self, slot, rears, errs, rlos, rlo2s, hpos, xs=None, y1s=None, hs=None, dirty=None):
        heap = self.rear_heap
        pos = hpos[slot]
        hpos[slot] = _HPOS_DEAD
        was_root = (pos == 0)
        last_idx = len(heap)-1
        last = heap.pop()
        self.live_count = last_idx
        if pos < last_idx:
            v_hi = rears[last]+errs[last]; v_lo = rlos[last]; v_lo2 = rlo2s[last]
            sifted_up = False
            if pos > 0:
                p = (pos-1) >> 1
                ps = heap[p]
                p_hi = rears[ps]+errs[ps]; p_lo = rlos[ps]; p_lo2 = rlo2s[ps]
                if v_hi > p_hi or (v_hi == p_hi and (v_lo > p_lo or (v_lo == p_lo and v_lo2 > p_lo2))):
                    while pos > 0:
                        p = (pos-1) >> 1
                        ps = heap[p]
                        p_hi = rears[ps]+errs[ps]; p_lo = rlos[ps]; p_lo2 = rlo2s[ps]
                        if p_hi > v_hi or (p_hi == v_hi and (p_lo > v_lo or (p_lo == v_lo and p_lo2 >= v_lo2))):
                            break
                        heap[pos] = ps
                        hpos[ps] = pos
                        pos = p
                    heap[pos] = last
                    hpos[last] = pos
                    sifted_up = True
            if not sifted_up:
                n = last_idx
                while True:
                    l = (pos << 1)+1
                    if l >= n:
                        break
                    r = l+1
                    best = l; best_s = heap[l]
                    b_hi = rears[best_s]+errs[best_s]; b_lo = rlos[best_s]; b_lo2 = rlo2s[best_s]
                    if r < n:
                        rs = heap[r]; r_hi = rears[rs]+errs[rs]; r_lo = rlos[rs]; r_lo2 = rlo2s[rs]
                        if r_hi > b_hi or (r_hi == b_hi and (r_lo > b_lo or (r_lo == b_lo and r_lo2 > b_lo2))):
                            best = r; best_s = rs; b_hi = r_hi; b_lo = r_lo; b_lo2 = r_lo2
                    if v_hi > b_hi or (v_hi == b_hi and (v_lo > b_lo or (v_lo == b_lo and v_lo2 >= b_lo2))):
                        break
                    heap[pos] = best_s
                    hpos[best_s] = pos
                    pos = best
                heap[pos] = last
                hpos[last] = pos
        rear_dropped = False
        if was_root:
            o_hi = self.max_rear_hi; o_lo = self.max_rear_lo; o_lo2 = self.max_rear_lo2
            if last_idx > 0:
                r0 = heap[0]
                n_hi = rears[r0]+errs[r0]; n_lo = rlos[r0]; n_lo2 = rlo2s[r0]
            else:
                n_hi = -math.inf; n_lo = 0.0; n_lo2 = 0.0
            self.max_rear_hi = n_hi; self.max_rear_lo = n_lo; self.max_rear_lo2 = n_lo2
            if n_hi < o_hi or (n_hi == o_hi and (n_lo < o_lo or (n_lo == o_lo and n_lo2 < o_lo2))):
                if dirty is not None:
                    rear_dropped = True
                else:
                    _h_propagate_rear_down(self.parent, o_hi, o_lo, o_lo2)
        mask = 0
        if last_idx == 0:
            mask = 15
        else:
            if slot == self.max_x_hi_slot: mask |= 1
            if slot == self.max_neg_x_lo_slot: mask |= 2
            if slot == self.max_y_hi_slot: mask |= 4
            if slot == self.max_neg_y_lo_slot: mask |= 8
        if mask or rear_dropped:
            if dirty is not None:
                if not self.bounds_dirty:
                    dirty.append(self)
                self.bounds_dirty |= mask | (16 if rear_dropped else 0)
            else:
                self.bounds_dirty |= mask
                self.refresh_bounds(xs, y1s, hs)

    def heap_rebuild(self, alive, xs, y1s, hs, rears, errs, rlos, rlo2s, hpos):
        heap = array("I", (s for s in self.slots if alive[s]))
        n = len(heap)
        for start in range((n >> 1)-1, -1, -1):
            pos = start
            s = heap[pos]
            v_hi = rears[s]+errs[s]; v_lo = rlos[s]; v_lo2 = rlo2s[s]
            while True:
                l = (pos << 1)+1
                if l >= n:
                    break
                r = l+1
                best = l; best_s = heap[l]
                b_hi = rears[best_s]+errs[best_s]; b_lo = rlos[best_s]; b_lo2 = rlo2s[best_s]
                if r < n:
                    rs = heap[r]; r_hi = rears[rs]+errs[rs]; r_lo = rlos[rs]; r_lo2 = rlo2s[rs]
                    if r_hi > b_hi or (r_hi == b_hi and (r_lo > b_lo or (r_lo == b_lo and r_lo2 > b_lo2))):
                        best = r; best_s = rs; b_hi = r_hi; b_lo = r_lo; b_lo2 = r_lo2
                if v_hi > b_hi or (v_hi == b_hi and (v_lo > b_lo or (v_lo == b_lo and v_lo2 >= b_lo2))):
                    break
                heap[pos] = best_s
                pos = best
            heap[pos] = s
        drift = OCCLUSION_TOP_DRIFT_HEIGHTS
        m_xh = m_nxl = m_yh = m_nyl = -math.inf
        s_xh = s_nxl = s_yh = s_nyl = -1
        for idx, s in enumerate(heap):
            hpos[s] = idx
            bx = xs[s]; by1 = y1s[s]; rb = drift*hs[s]
            xh = bx+rb; nxl = rb-bx; yh = by1+rb; nyl = rb-by1
            if xh > m_xh: m_xh = xh; s_xh = s
            if nxl > m_nxl: m_nxl = nxl; s_nxl = s
            if yh > m_yh: m_yh = yh; s_yh = s
            if nyl > m_nyl: m_nyl = nyl; s_nyl = s
        self.rear_heap = heap
        self.live_count = n
        if n > 0:
            r0 = heap[0]
            self.max_rear_hi = rears[r0]+errs[r0]; self.max_rear_lo = rlos[r0]; self.max_rear_lo2 = rlo2s[r0]
        else:
            self.max_rear_hi = -math.inf; self.max_rear_lo = 0.0; self.max_rear_lo2 = 0.0
        self.max_x_hi = m_xh; self.max_x_hi_slot = s_xh
        self.max_neg_x_lo = m_nxl; self.max_neg_x_lo_slot = s_nxl
        self.max_y_hi = m_yh; self.max_y_hi_slot = s_yh
        self.max_neg_y_lo = m_nyl; self.max_neg_y_lo_slot = s_nyl
        self.bounds_dirty = 0


class _HNode:
    __slots__ = ("children", "parent", "max_rear_hi", "max_rear_lo", "max_rear_lo2", "min_h", "max_h", "max_y2", "max_x",
                 "max_x_hi", "max_neg_x_lo", "max_y_hi", "max_neg_y_lo")

    def __init__(self, children, parent=None):
        self.children = children
        self.parent = parent
        self.max_rear_hi = -math.inf; self.max_rear_lo = 0.0; self.max_rear_lo2 = 0.0
        self.min_h = math.inf
        self.max_h = self.max_y2 = self.max_x = -math.inf
        self.max_x_hi = self.max_neg_x_lo = self.max_y_hi = self.max_neg_y_lo = -math.inf
        self.refresh()

    def refresh(self):
        ch = self.children
        c0 = ch[0]; c1 = ch[-1]
        self.min_h = c0.h[0] if type(c0) is _HBlock else c0.min_h
        if type(c1) is _HBlock:
            self.max_h = c1.h[-1]; self.max_y2 = c1.y2[-1]; self.max_x = c1.x[-1]
        else:
            self.max_h = c1.max_h; self.max_y2 = c1.max_y2; self.max_x = c1.max_x
        m_hi = -math.inf; m_lo = 0.0; m_lo2 = 0.0
        m_xh = m_nxl = m_yh = m_nyl = -math.inf
        for c in ch:
            c.parent = self
            chi = c.max_rear_hi; clo = c.max_rear_lo; clo2 = c.max_rear_lo2
            if chi > m_hi or (chi == m_hi and (clo > m_lo or (clo == m_lo and clo2 > m_lo2))):
                m_hi = chi; m_lo = clo; m_lo2 = clo2
            if c.max_x_hi > m_xh: m_xh = c.max_x_hi
            if c.max_neg_x_lo > m_nxl: m_nxl = c.max_neg_x_lo
            if c.max_y_hi > m_yh: m_yh = c.max_y_hi
            if c.max_neg_y_lo > m_nyl: m_nyl = c.max_neg_y_lo
        self.max_rear_hi = m_hi; self.max_rear_lo = m_lo; self.max_rear_lo2 = m_lo2
        self.max_x_hi = m_xh; self.max_neg_x_lo = m_nxl
        self.max_y_hi = m_yh; self.max_neg_y_lo = m_nyl


def _step_retire(queue, budget):
    """Incrementally unlink and release retired index jobs in FIFO order within a
    per-frame budget so large teardown stays O(1) and no retired job can starve."""
    worked = 0
    while queue and worked < budget:
        stack = queue[0]
        if not stack:
            queue.popleft()
            continue
        item = stack.pop()
        t = type(item)
        if t is _RNode:
            ch = item.children
            if ch is not None:
                stack.extend(ch)
                item.children = None
            item.idxs = None
            worked += 1
        elif t is _HNode:
            ch = item.children
            if ch is not None:
                for c in ch:
                    c.parent = None
                stack.extend(ch)
                item.children = None
            item.parent = None
            worked += 1
        elif t is _HBlock:
            item.parent = None
            item.next_blk = None
            worked += 1
        elif t is list:
            n = len(item)
            take = max(1, (budget-worked) >> 2)
            if n <= take:
                item.clear()
                worked += (n << 2) or 1
            else:
                del item[-take:]
                worked += take << 2
                stack.append(item)
        elif t is tuple:
            tag = item[0]
            if tag == 0:
                lst = item[1]
                for el in lst:
                    if el is not None:
                        if type(el) is list:
                            stack.append((0, el))
                        worked += 1
                lst.clear()
                worked += 1
            elif tag == 2:
                bufs = item[1]
                while bufs and worked < budget:
                    bufs.pop()
                    worked += 128
                if bufs:
                    stack.append(item)
            elif tag == 3:
                d = item[1]
                while d and worked < budget:
                    d.popitem()
                    worked += 1
                if d:
                    stack.append(item)
            else:
                sub = item[1]
                step = _step_retire(sub, budget-worked)
                worked += step if step else 1
                if sub:
                    stack.append(item)
        else:
            worked += 1
        if not stack:
            queue.popleft()
    return worked


class _ClipBaselines:
    """Lossless compact-array indexed stationary clip baselines for one active track.

    - Baselines live in contiguous ``array('d')`` columns plus a 1-byte ``_alive``
      bitmap (minimal memory and O(1) teardown).
    - Four ``_BoundBlocks`` boundary indexes perform O(k log N) bulk-prefix partial
      pruning while live 4-axis envelopes on ``_HNode`` perform O(1) whole-track
      eviction even after sparse cross-axis pruning.
    - A self-balancing 2D (x, y1) R-tree (``_RNode``) carrying height, rear, and
      drift envelopes answers dominance/duplicate checks and live dominance
      eviction with strict O(log N) split-only inserts (zero subtree rebuilds).
    - An augmented B-tree of sorted ``(h, y2, x)`` ``_HBlock`` leaves with
      eager-delete indexed per-block rear max-heaps maintains exact live
      3-term rear maxima across kills and splits, answering crossing queries
      in descending ``(h, y2)`` order in O(log N) with zero cutoff dead zone.
    """
    __slots__ = ("_x", "_y1", "_y2", "_h", "_rear", "_err", "_rlo", "_rlo2", "_alive", "count",
                 "_root",
                 "_x_hi", "_x_lo", "_y_hi", "_y_lo",
                 "_hroot", "_blk_by_id", "_slot_bid", "_hpos", "_rebuild",
                 "_dom_sweep", "_dom_queue", "_dom_pending", "_dom_cache",
                 "_cover_probe", "_cover_pending", "_cover_count", "_cover_scan", "_sweep_turn",
                 "_std_geom", "_exact_rear", "_retired", "_own_retired")

    def __init__(self, *, exact_rear=False, retire=None):
        self._x = array("d")
        self._y1 = array("d")
        self._y2 = array("d")
        self._h = array("d")
        self._rear = array("d")
        self._err = array("d")
        self._rlo = array("d")
        self._rlo2 = array("d")
        self._alive = bytearray()
        self.count = 0
        self._root = None
        self._x_hi = _BoundBlocks()
        self._x_lo = _BoundBlocks()
        self._y_hi = _BoundBlocks()
        self._y_lo = _BoundBlocks()
        self._hroot = None
        self._blk_by_id = []
        self._slot_bid = array("I")
        self._hpos = array("H")
        self._rebuild = None
        self._dom_sweep = None
        self._dom_queue = _SlotFIFO()
        self._dom_pending = bytearray()
        self._dom_cache = []
        self._cover_probe = None
        self._cover_pending = bytearray()
        self._cover_count = 0
        self._cover_scan = 0
        self._sweep_turn = False
        self._std_geom = True
        self._exact_rear = exact_rear
        self._own_retired = retire is None
        self._retired = deque() if retire is None else retire

    def __bool__(self):
        return self.count > 0

    def __len__(self):
        return self.count

    @property
    def _blocks(self):
        node = self._hroot
        if node is None:
            return []
        while type(node) is _HNode:
            node = node.children[0]
        out = []
        while node is not None:
            out.append(node)
            node = node.next_blk
        return out

    def retire_to(self, sink):
        if (self.count == 0 and len(self._x) == 0 and self._rebuild is None and
                self._root is None and self._hroot is None and not self._blk_by_id and
                (self._retired is sink or not self._retired)):
            return
        if (len(self._x) <= 32 and len(self._blk_by_id) <= 2 and
                self._rebuild is None and len(self._dom_queue) <= 256 and
                (self._retired is sink or not self._retired)):
            if type(self._hroot) is _HNode:
                self._hroot.children = None
            for blk in self._blk_by_id:
                blk.parent = None
                blk.next_blk = None
            self._blk_by_id.clear()
            self._hroot = None
            self._root = None
            self._x_hi.clear(); self._x_lo.clear(); self._y_hi.clear(); self._y_lo.clear()
            self._dom_queue.clear()
        else:
            job = []
            if self._rebuild is not None:
                shadow, _, _, mapping, _, q_shadow = self._rebuild
                self._rebuild = None
                shadow.retire_to(sink)
                if q_shadow is not None:
                    q_shadow.retire_to(job)
                if len(mapping) > 64:
                    job.append((3, mapping))
            if self._blk_by_id:
                job.append(self._blk_by_id)
                self._blk_by_id = []
            if self._hroot is not None:
                job.append(self._hroot)
                self._hroot = None
            if self._root is not None:
                job.append(self._root)
                self._root = None
            for bb in (self._x_hi, self._x_lo, self._y_hi, self._y_lo):
                if bb.blocks:
                    job.append(bb.blocks)
                    bb.blocks = []
                bb.max_vals = array("d")
            self._dom_queue.retire_to(job)
            if self._retired is not sink and self._retired:
                job.append((1, self._retired))
                self._retired = deque() if self._own_retired else sink
            if len(self._x) >= 8192:
                job.append((2, [self._x, self._y1, self._y2, self._h, self._rear, self._err,
                                self._rlo, self._rlo2, self._alive, self._slot_bid, self._hpos,
                                self._dom_pending, self._cover_pending]))
            if job:
                sink.append(job)
        self._x = array("d"); self._y1 = array("d"); self._y2 = array("d")
        self._h = array("d"); self._rear = array("d"); self._err = array("d")
        self._rlo = array("d"); self._rlo2 = array("d")
        self._alive = bytearray()
        self.count = 0
        self._slot_bid = array("I"); self._hpos = array("H")
        self._dom_sweep = None
        self._dom_pending = bytearray()
        self._dom_cache.clear()
        self._cover_probe = None
        self._cover_pending = bytearray()
        self._cover_count = 0
        self._cover_scan = 0
        self._sweep_turn = False
        self._std_geom = True

    def clear(self):
        self.retire_to(self._retired)

    def _h_locate(self, h, y2, x):
        node = self._hroot
        if node is None:
            return None
        while type(node) is _HNode:
            ch = node.children
            chosen = ch[-1]
            for c in ch:
                if type(c) is _HBlock:
                    if (c.h[-1], c.y2[-1], c.x[-1]) >= (h, y2, x):
                        chosen = c
                        break
                elif (c.max_h, c.max_y2, c.max_x) >= (h, y2, x):
                    chosen = c
                    break
            node = chosen
        return node

    def _on_hblock_split(self, blk, right):
        rb = self._rebuild
        if rb is None or right.max_rear_hi == -math.inf:
            return
        cstack = rb[1]
        for idx_f, item in enumerate(cstack):
            if item is blk:
                cstack.insert(idx_f+1, right)
                return

    def _on_hnode_split(self, parent, sibling):
        rb = self._rebuild
        if rb is None or sibling.max_rear_hi == -math.inf:
            return
        cstack = rb[1]
        for idx_f, item in enumerate(cstack):
            if item is parent:
                cstack.insert(idx_f+1, sibling)
                return

    def _h_insert_sibling(self, left, right):
        parent = left.parent
        if parent is None:
            self._hroot = _HNode([left, right])
            return
        ch = parent.children
        idx = ch.index(left)
        ch.insert(idx+1, right)
        right.parent = parent
        if len(ch) <= _HNODE_MAX:
            parent.refresh()
            cur = parent.parent
            while cur is not None:
                cur.refresh()
                cur = cur.parent
            return
        mid = len(ch)//2
        parent.children = ch[:mid]
        parent.refresh()
        sibling = _HNode(ch[mid:])
        self._on_hnode_split(parent, sibling)
        self._h_insert_sibling(parent, sibling)

    @staticmethod
    def _find_in_block(blk, h, y2, x):
        bh = blk.h
        lo = bisect.bisect_left(bh, h)
        if lo < len(bh) and bh[lo] == h:
            hi = bisect.bisect_right(bh, h, lo)
            ylo = bisect.bisect_left(blk.y2, y2, lo, hi)
            if ylo < hi and blk.y2[ylo] == y2:
                yhi = bisect.bisect_right(blk.y2, y2, ylo, hi)
                return bisect.bisect_left(blk.x, x, ylo, yhi)
            return ylo
        return lo

    def _has_exact(self, bx, by1, by2, bh):
        if self._hroot is None or self.count == 0:
            return False
        blk = self._h_locate(bh, by2, bx)
        alive = self._alive; y1s = self._y1
        while blk is not None:
            h_arr = blk.h
            lo = bisect.bisect_left(h_arr, bh)
            if lo == len(h_arr) or h_arr[lo] != bh:
                return False
            hi = bisect.bisect_right(h_arr, bh, lo)
            y2_arr = blk.y2
            ylo = bisect.bisect_left(y2_arr, by2, lo, hi)
            if ylo == hi or y2_arr[ylo] != by2:
                return False
            yhi = bisect.bisect_right(y2_arr, by2, ylo, hi)
            x_arr = blk.x
            k = bisect.bisect_left(x_arr, bx, ylo, yhi)
            slots = blk.slots
            while k < yhi and x_arr[k] == bx:
                s = slots[k]
                if alive[s] and y1s[s] == by1:
                    return True
                k += 1
            if k < len(h_arr):
                return False
            blk = blk.next_blk
        return False

    def _blocks_add(self, h, y2, x, y1, rear, err, slot):
        self._hpos.append(_HPOS_DEAD)
        if self._hroot is None:
            bid = len(self._blk_by_id)
            blk = _HBlock(array("d", (h,)), array("d", (y2,)), array("d", (x,)), array("I", (slot,)), bid=bid)
            self._blk_by_id.append(blk)
            self._hroot = blk
            self._slot_bid.append(bid)
            blk.heap_push(slot, x, y1, h, self._rear, self._err, self._rlo, self._rlo2, self._hpos)
            return
        blk = self._h_locate(h, y2, x)
        bh = blk.h; by2 = blk.y2; bx_arr = blk.x
        i = self._find_in_block(blk, h, y2, x)
        if (i < len(bh) and bh[i] == h and by2[i] == y2 and bx_arr[i] == x and
                not self._alive[blk.slots[i]] and self._y1[blk.slots[i]] == y1):
            blk.slots[i] = slot
            self._slot_bid.append(blk.bid)
            blk.heap_push(slot, x, y1, h, self._rear, self._err, self._rlo, self._rlo2, self._hpos)
            return
        bh.insert(i, h); by2.insert(i, y2); bx_arr.insert(i, x); blk.slots.insert(i, slot)
        self._slot_bid.append(blk.bid)
        blk.heap_push(slot, x, y1, h, self._rear, self._err, self._rlo, self._rlo2, self._hpos)
        if i == 0:
            cur = blk.parent
            while cur is not None and h < cur.min_h:
                cur.min_h = h
                cur = cur.parent
        if i == len(bh)-1:
            cur = blk.parent
            while cur is not None and (h, y2, x) > (cur.max_h, cur.max_y2, cur.max_x):
                cur.max_h = h; cur.max_y2 = y2; cur.max_x = x
                cur = cur.parent
        if len(bh) > _HBLOCK_MAX:
            mid = len(bh)//2
            alive = self._alive; slot_bid = self._slot_bid
            xs = self._x; y1s = self._y1; hs = self._h
            rears = self._rear; errs = self._err; rlos = self._rlo; rlo2s = self._rlo2; hpos = self._hpos
            rbid = len(self._blk_by_id)
            right = _HBlock(bh[mid:], by2[mid:], bx_arr[mid:], blk.slots[mid:], next_blk=blk.next_blk, bid=rbid)
            self._blk_by_id.append(right)
            blk.next_blk = right
            for s in right.slots:
                slot_bid[s] = rbid
            del bh[mid:]; del by2[mid:]; del bx_arr[mid:]; del blk.slots[mid:]
            blk.heap_rebuild(alive, xs, y1s, hs, rears, errs, rlos, rlo2s, hpos)
            right.heap_rebuild(alive, xs, y1s, hs, rears, errs, rlos, rlo2s, hpos)
            self._on_hblock_split(blk, right)
            self._h_insert_sibling(blk, right)

    def _kill_slot(self, i):
        alive = self._alive
        if not alive[i]:
            return
        alive[i] = 0
        self._blk_by_id[self._slot_bid[i]].heap_remove(i, self._rear, self._err, self._rlo, self._rlo2, self._hpos,
                                                       self._x, self._y1, self._h)
        self.count -= 1
        self._dom_pending[i] = 0
        if self._cover_pending[i]:
            self._cover_pending[i] = 0
            self._cover_count -= 1
        if self._rebuild is not None:
            shadow, _, _, mapping, _, q_shadow = self._rebuild
            new_i = mapping.pop(i, None)
            if new_i is not None and shadow._alive[new_i]:
                shadow._alive[new_i] = 0
                shadow._blk_by_id[shadow._slot_bid[new_i]].heap_remove(
                    new_i, shadow._rear, shadow._err, shadow._rlo, shadow._rlo2, shadow._hpos,
                    shadow._x, shadow._y1, shadow._h)
                shadow.count -= 1
                shadow._dom_pending[new_i] = 0
                if shadow._cover_pending[new_i]:
                    shadow._cover_pending[new_i] = 0
                    shadow._cover_count -= 1
                if shadow.count == 0:
                    self._rebuild = None
                    shadow.retire_to(self._retired)
                    extra = []
                    if q_shadow is not None:
                        q_shadow.retire_to(extra)
                    if len(mapping) > 64:
                        extra.append((3, mapping))
                    if extra:
                        self._retired.append(extra)

    def _cache_dom(self, slot):
        dc = self._dom_cache
        if slot in dc:
            dc.remove(slot)
        elif len(dc) >= 8:
            dc.pop(0)
        dc.append(slot)

    def _covered_by_hot(self, bx, by1, by2, bh, brear, berr, brlo=0.0, brlo2=0.0):
        alive = self._alive
        xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
        rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
        if self._dom_sweep is not None:
            ds = self._dom_sweep[0]
            if 0 <= ds < len(alive) and alive[ds] and _dominates_coords(
                    xs[ds], y1s[ds], y2s[ds], hs[ds], rears[ds], errs[ds],
                    bx, by1, by2, bh, brear, berr, rlos[ds], brlo, rlo2s[ds], brlo2):
                return True
        dc = self._dom_cache
        if not dc:
            return False
        for idx in range(len(dc)-1, -1, -1):
            s = dc[idx]
            if s >= len(alive) or not alive[s]:
                del dc[idx]
            elif _dominates_coords(xs[s], y1s[s], y2s[s], hs[s], rears[s], errs[s],
                                   bx, by1, by2, bh, brear, berr, rlos[s], brlo, rlo2s[s], brlo2):
                return True
        return False

    def _insert_raw(self, bx, by1, by2, bh, brear, berr, brlo=0.0, brlo2=0.0):
        if self._exact_rear:
            if self._std_geom and (by1 != by2-bh or berr != 0.0):
                self._std_geom = False
        elif self._std_geom and (by1 != by2-bh or brear != by2-OCCLUSION_BOTTOM_SHARE*bh or berr != _rear_err(by2, bh)):
            self._std_geom = False
        slot = len(self._x)
        self._x.append(bx); self._y1.append(by1); self._y2.append(by2)
        self._h.append(bh); self._rear.append(brear); self._err.append(berr)
        self._rlo.append(brlo); self._rlo2.append(brlo2)
        self._alive.append(1)
        self._dom_pending.append(0)
        self._cover_pending.append(0)
        self.count += 1
        rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        self._x_hi.push(bx+rb, slot); self._x_lo.push(-(bx-rb), slot)
        self._y_hi.push(by1+rb, slot); self._y_lo.push(-(by1-rb), slot)
        self._blocks_add(bh, by2, bx, by1, brear, berr, slot)
        self._rtree_insert(slot, bx, by1, bh, brear, berr)
        return slot

    def _step_compact(self, budget):
        rb = self._rebuild
        shadow, cstack, end, mapping, q_scan, q_shadow = rb
        alive = self._alive
        xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
        rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
        copied = 0
        if cstack:
            step_limit = max(64, budget << 3)
            steps = 0
            while cstack and copied < budget and steps < step_limit:
                item = cstack.pop()
                t = type(item)
                if t is _HNode:
                    steps += 1
                    if item.max_rear_hi == -math.inf:
                        continue
                    for c in reversed(item.children):
                        if c.max_rear_hi != -math.inf:
                            cstack.append(c)
                elif t is _HBlock:
                    steps += 1
                    if item.max_rear_hi == -math.inf:
                        continue
                    snap = array("I", sorted(s for s in item.rear_heap if s < end))
                    if snap:
                        cstack.append((snap, 0))
                else:
                    snap, idx = item
                    n_snap = len(snap)
                    while idx < n_snap and copied < budget:
                        s = snap[idx]
                        idx += 1
                        if alive[s] and s not in mapping:
                            new_i = shadow._insert_raw(xs[s], y1s[s], y2s[s], hs[s], rears[s], errs[s], rlos[s], rlo2s[s])
                            mapping[s] = new_i
                            if self._cover_pending[s]:
                                shadow._cover_pending[new_i] = 1
                                shadow._cover_count += 1
                            copied += 1
                    if idx < n_snap:
                        cstack.append((snap, idx))
            if cstack or copied >= budget:
                return copied or 1
        q_dq = self._dom_queue
        q_head = q_dq.base
        if q_shadow is None:
            q_scan = q_head+len(q_dq)
            q_shadow = _SlotFIFO()
            rb[4] = q_scan
            rb[5] = q_shadow
        while q_scan > q_head and copied < budget:
            q_scan -= 1
            q = q_dq[q_scan-q_head]
            nq = mapping.get(q, -1)
            if nq >= 0 and shadow._alive[nq] and not shadow._dom_pending[nq]:
                shadow._dom_pending[nq] = 1
            else:
                nq = -1
            q_shadow.appendleft(nq)
            copied += 1
        if q_scan > q_head:
            rb[4] = q_scan
            return copied
        job = []
        if self._blk_by_id:
            job.append(self._blk_by_id)
        if self._hroot is not None:
            job.append(self._hroot)
        if self._root is not None:
            job.append(self._root)
        for bb in (self._x_hi, self._x_lo, self._y_hi, self._y_lo):
            if bb.blocks:
                job.append(bb.blocks)
        self._dom_queue.retire_to(job)
        if len(self._x) >= 8192:
            job.append((2, [self._x, self._y1, self._y2, self._h, self._rear, self._err,
                            self._rlo, self._rlo2, self._alive, self._slot_bid, self._hpos,
                            self._dom_pending, self._cover_pending]))
        if len(mapping) > 64:
            job.append((3, mapping))
        if job:
            self._retired.append(job)
        if self._dom_sweep is not None:
            new_dom = mapping.get(self._dom_sweep[0], -1)
            if new_dom >= 0 and shadow._alive[new_dom]:
                self._dom_sweep = (new_dom, -2, len(shadow._x))
                shadow._dom_pending[new_dom] = 1
            else:
                self._dom_sweep = None
        if self._dom_cache:
            self._dom_cache = [nq for s in self._dom_cache
                               if (nq := mapping.get(s)) is not None and shadow._alive[nq]]
        if self._cover_probe is not None:
            new_c = mapping.get(self._cover_probe[0], -1)
            self._cover_probe = ((new_c, None, len(shadow._x))
                                 if (new_c >= 0 and shadow._alive[new_c]) else None)
        self._dom_queue = q_shadow
        self._dom_pending = shadow._dom_pending
        self._cover_pending = shadow._cover_pending
        self._cover_count = shadow._cover_count
        self._cover_scan = 0
        self._x, self._y1, self._y2 = shadow._x, shadow._y1, shadow._y2
        self._h, self._rear, self._err = shadow._h, shadow._rear, shadow._err
        self._rlo, self._rlo2 = shadow._rlo, shadow._rlo2
        self._alive, self.count, self._root = shadow._alive, shadow.count, shadow._root
        self._x_hi, self._x_lo = shadow._x_hi, shadow._x_lo
        self._y_hi, self._y_lo = shadow._y_hi, shadow._y_lo
        self._hroot, self._blk_by_id, self._slot_bid, self._hpos = (
            shadow._hroot, shadow._blk_by_id, shadow._slot_bid, shadow._hpos)
        self._std_geom = shadow._std_geom
        self._rebuild = None
        return copied or 1

    def has_dom_sweep(self):
        return (self._dom_sweep is not None or bool(self._dom_queue) or
                self._cover_probe is not None or self._cover_count > 0)

    def _queue_cover_probe(self, slot):
        if self._cover_probe is None and self._cover_count == 0:
            self._cover_probe = (slot, None, len(self._alive))
            return
        if not self._cover_pending[slot]:
            self._cover_pending[slot] = 1
            self._cover_count += 1
            if slot < self._cover_scan:
                self._cover_scan = slot
            if self._rebuild is not None:
                ns = self._rebuild[3].get(slot, -1)
                if ns >= 0 and self._rebuild[0]._alive[ns] and not self._rebuild[0]._cover_pending[ns]:
                    self._rebuild[0]._cover_pending[ns] = 1
                    self._rebuild[0]._cover_count += 1

    def _queue_dom_sweep(self, slot, bx, by1, by2, bh, brear, berr, brlo=0.0, brlo2=0.0):
        alive = self._alive
        dp = self._dom_pending
        if slot < 0 or slot >= len(alive) or not alive[slot] or dp[slot]:
            return
        xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
        rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
        if self._dom_sweep is not None:
            ds = self._dom_sweep[0]
            if ds < 0 or ds >= len(alive) or not alive[ds]:
                if 0 <= ds < len(dp):
                    dp[ds] = 0
                self._dom_sweep = None
            elif _dominates_coords(xs[ds], y1s[ds], y2s[ds], hs[ds], rears[ds], errs[ds],
                                   bx, by1, by2, bh, brear, berr, rlos[ds], brlo, rlo2s[ds], brlo2):
                return
            elif _dominates_coords(bx, by1, by2, bh, brear, berr,
                                   xs[ds], y1s[ds], y2s[ds], hs[ds], rears[ds], errs[ds],
                                   brlo, rlos[ds], brlo2, rlo2s[ds]):
                self._kill_slot(ds)
                self._dom_sweep = None
        q_dq = self._dom_queue
        rb = self._rebuild
        if q_dq:
            tail = []
            inspected = 0
            dominated = False
            while q_dq and inspected < 16:
                q = q_dq.pop()
                sq = None
                if rb is not None and rb[5] is not None:
                    if rb[5]:
                        sq = rb[5].pop()
                    else:
                        rb[4] -= 1
                inspected += 1
                if q < 0 or q >= len(alive) or not alive[q]:
                    if 0 <= q < len(dp):
                        dp[q] = 0
                    if rb is not None and rb[5] is not None and sq is not None and sq >= 0:
                        rb[0]._dom_pending[sq] = 0
                    continue
                if _dominates_coords(xs[q], y1s[q], y2s[q], hs[q], rears[q], errs[q],
                                     bx, by1, by2, bh, brear, berr, rlos[q], brlo, rlo2s[q], brlo2):
                    tail.append((q, sq))
                    dominated = True
                    break
                if _dominates_coords(bx, by1, by2, bh, brear, berr,
                                     xs[q], y1s[q], y2s[q], hs[q], rears[q], errs[q],
                                     brlo, rlos[q], brlo2, rlo2s[q]):
                    self._kill_slot(q)
                    rb = self._rebuild
                    continue
                tail.append((q, sq))
            if tail:
                for q, sq in reversed(tail):
                    q_dq.append(q)
                    if rb is not None and rb[5] is not None:
                        if sq is None:
                            rb[4] += 1
                        else:
                            rb[5].append(sq)
            if dominated:
                return
        dp[slot] = 1
        if self._dom_sweep is None and not q_dq:
            self._dom_sweep = (slot, -2, len(alive))
        else:
            q_dq.append(slot)
            if rb is not None and rb[5] is not None:
                nq = rb[3].get(slot, -1)
                if nq >= 0 and rb[0]._alive[nq] and not rb[0]._dom_pending[nq]:
                    rb[0]._dom_pending[nq] = 1
                else:
                    nq = -1
                rb[5].append(nq)

    def _on_leaf_insert(self, leaf, k):
        ds = self._dom_sweep
        if ds is not None and isinstance(ds[1], list) and ds[1]:
            top = ds[1][-1]
            if top[0] is leaf and top[1] is not None and k <= top[1]:
                top[1] += 1
        cp = self._cover_probe
        if cp is not None and cp[1] is not None and cp[1]:
            top = cp[1][-1]
            if top[0] is leaf and top[1] is not None and k < top[1]:
                top[1] += 1

    def _on_leaf_compact(self, leaf, old_idxs, new_idxs):
        ds = self._dom_sweep
        if ds is not None and isinstance(ds[1], list) and ds[1]:
            top = ds[1][-1]
            if top[0] is leaf and top[1] is not None:
                if not new_idxs:
                    top[1] = -1
                else:
                    nset = set(new_idxs)
                    top[1] = sum(1 for s in old_idxs[:top[1]+1] if s in nset)-1
        cp = self._cover_probe
        if cp is not None and cp[1] is not None and cp[1]:
            top = cp[1][-1]
            if top[0] is leaf and top[1] is not None:
                if not new_idxs:
                    top[1] = 0
                else:
                    nset = set(new_idxs)
                    top[1] = sum(1 for s in old_idxs[:top[1]] if s in nset)

    def _on_leaf_split(self, leaf, sibling, old_idxs):
        alive = self._alive
        ds = self._dom_sweep
        if ds is not None and isinstance(ds[1], list) and ds[1]:
            stack, sweep_end = ds[1], ds[2]
            for idx_f, frame in enumerate(stack):
                if frame[0] is leaf:
                    if frame[1] is None:
                        if sibling.min_slot < sweep_end:
                            stack.insert(idx_f+1, [sibling, None])
                    else:
                        rem = array("I", (s for s in old_idxs[:frame[1]+1] if s < sweep_end and alive[s]))
                        stack[idx_f] = [rem, len(rem)-1]
                    break
        cp = self._cover_probe
        if cp is not None and cp[1] is not None and cp[1]:
            stack, probe_end = cp[1], cp[2]
            for idx_f, frame in enumerate(stack):
                if frame[0] is leaf:
                    if frame[1] is None:
                        if sibling.min_slot < probe_end:
                            stack.insert(idx_f+1, [sibling, None])
                    else:
                        rem = array("I", (s for s in old_idxs[frame[1]:] if s < probe_end and alive[s]))
                        stack[idx_f] = [rem, 0]
                    break

    def _on_internal_split(self, parent, sibling):
        ds = self._dom_sweep
        if ds is not None and isinstance(ds[1], list) and ds[1]:
            stack, sweep_end = ds[1], ds[2]
            if sibling.min_slot < sweep_end:
                for idx_f, frame in enumerate(stack):
                    if frame[0] is parent:
                        stack.insert(idx_f+1, [sibling, None])
                        break
        cp = self._cover_probe
        if cp is not None and cp[1] is not None and cp[1]:
            stack, probe_end = cp[1], cp[2]
            if sibling.min_slot < probe_end:
                for idx_f, frame in enumerate(stack):
                    if frame[0] is parent:
                        stack.insert(idx_f+1, [sibling, None])
                        break

    def _step_evict_sweep(self, budget):
        if budget <= 0:
            return 0
        alive = self._alive
        dp = self._dom_pending
        std_geom = self._std_geom
        xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
        rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
        checked = 0
        while checked < budget:
            iter_start = checked
            while self._dom_sweep is None and self._dom_queue and checked < budget:
                q_head = self._dom_queue.base
                cand = self._dom_queue.popleft()
                if self._rebuild is not None and self._rebuild[5] is not None and q_head == self._rebuild[4]:
                    sq = self._rebuild[5].popleft()
                    self._rebuild[4] += 1
                    if sq >= 0:
                        self._rebuild[0]._dom_pending[sq] = 0
                if 0 <= cand < len(alive) and alive[cand]:
                    self._dom_sweep = (cand, -2, len(alive))
                else:
                    if 0 <= cand < len(dp):
                        dp[cand] = 0
                    checked += 1
            if self._dom_sweep is None:
                break
            dom_slot, stack, sweep_end = self._dom_sweep
            if dom_slot < 0 or dom_slot >= len(alive) or not alive[dom_slot]:
                if 0 <= dom_slot < len(dp):
                    dp[dom_slot] = 0
                self._dom_sweep = None
                if checked == iter_start:
                    checked += 1
                continue
            ax, ay1, ay2, ah, arear, aerr, arlo, arlo2 = (
                xs[dom_slot], y1s[dom_slot], y2s[dom_slot],
                hs[dom_slot], rears[dom_slot], errs[dom_slot], rlos[dom_slot], rlo2s[dom_slot])
            arear_lo = arear-aerr
            ra = OCCLUSION_TOP_DRIFT_HEIGHTS*ah
            ax_lo, ax_hi, ay_lo, ay_hi = ax-ra, ax+ra, ay1-ra, ay1+ra
            if stack == -2:
                checked += 1
                root = self._root
                if (root is None or root.min_slot >= sweep_end or
                        not self._node_may_be_dominated(root, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom)):
                    dp[dom_slot] = 0
                    self._dom_sweep = None
                    continue
                if root.idxs is not None:
                    stack = [[root, None]]
                else:
                    stack = [[c, None] for c in root.children
                             if c.min_slot < sweep_end and
                             self._node_may_be_dominated(c, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom)]
                    if not stack:
                        dp[dom_slot] = 0
                        self._dom_sweep = None
                        continue
                self._dom_sweep = (dom_slot, stack, sweep_end)
                if checked >= budget:
                    break
            while stack and checked < budget:
                frame = stack[-1]
                node = frame[0]
                if type(node) is _RNode:
                    if (node.min_slot >= sweep_end or
                            not self._node_may_be_dominated(node, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom)):
                        stack.pop()
                        checked += 1
                        continue
                    if node.idxs is None:
                        stack.pop()
                        checked += 1
                        for c in node.children:
                            if (c.min_slot < sweep_end and
                                    self._node_may_be_dominated(c, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom)):
                                stack.append([c, None])
                        continue
                    idxs = node.idxs
                    idx = len(idxs)-1 if frame[1] is None else frame[1]
                    if idx < 0:
                        stack.pop()
                        checked += 1
                        continue
                    ev_leaf = False
                    while idx >= 0 and checked < budget:
                        s = idxs[idx]
                        idx -= 1
                        checked += 1
                        if s < sweep_end and s != dom_slot and alive[s]:
                            if (hs[s] >= ah if std_geom else hs[s] > ah):
                                idx = -1
                                break
                            if _dominates_coords(ax, ay1, ay2, ah, arear, aerr,
                                                 xs[s], y1s[s], y2s[s], hs[s], rears[s], errs[s],
                                                 arlo, rlos[s], arlo2, rlo2s[s]):
                                self._kill_slot(s)
                                self._cache_dom(dom_slot)
                                ev_leaf = True
                    if idx < 0:
                        stack.pop()
                        if ev_leaf:
                            live_leaf = [i for i in idxs if alive[i]]
                            if live_leaf:
                                new_idxs = array("I", live_leaf)
                                node.idxs = new_idxs
                                node.refresh_leaf(xs, y1s, hs, rears, errs)
                                self._on_leaf_compact(node, idxs, new_idxs)
                    else:
                        frame[1] = idx
                else:
                    idx = frame[1]
                    if idx < 0:
                        stack.pop()
                        checked += 1
                        continue
                    while idx >= 0 and checked < budget:
                        s = node[idx]
                        idx -= 1
                        checked += 1
                        if s < sweep_end and s != dom_slot and alive[s]:
                            if _dominates_coords(ax, ay1, ay2, ah, arear, aerr,
                                                 xs[s], y1s[s], y2s[s], hs[s], rears[s], errs[s],
                                                 arlo, rlos[s], arlo2, rlo2s[s]):
                                self._kill_slot(s)
                                self._cache_dom(dom_slot)
                    if idx < 0:
                        stack.pop()
                    else:
                        frame[1] = idx
            if not stack:
                dp[dom_slot] = 0
                self._dom_sweep = None
                if checked == iter_start:
                    checked += 1
        return checked

    def _step_cover_probe(self, budget):
        if budget <= 0:
            return 0
        alive = self._alive
        cp = self._cover_pending
        std_geom = self._std_geom
        xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
        rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
        checked = 0
        while checked < budget:
            iter_start = checked
            while self._cover_probe is None and self._cover_count > 0 and checked < budget:
                cand = cp.find(1, self._cover_scan)
                if cand == -1:
                    self._cover_count = 0
                    self._cover_scan = len(cp)
                    break
                cp[cand] = 0
                self._cover_count -= 1
                self._cover_scan = cand+1
                if self._rebuild is not None:
                    ns = self._rebuild[3].get(cand, -1)
                    if ns >= 0 and self._rebuild[0]._cover_pending[ns]:
                        self._rebuild[0]._cover_pending[ns] = 0
                        self._rebuild[0]._cover_count -= 1
                if alive[cand]:
                    self._cover_probe = (cand, None, len(alive))
                else:
                    checked += 1
            if self._cover_probe is None:
                break
            cand, stack, probe_end = self._cover_probe
            if cand < 0 or cand >= len(alive) or not alive[cand]:
                self._cover_probe = None
                if checked == iter_start:
                    checked += 1
                continue
            bx, by1, by2, bh, brear, berr, brlo, brlo2 = (
                xs[cand], y1s[cand], y2s[cand],
                hs[cand], rears[cand], errs[cand], rlos[cand], rlo2s[cand])
            brear_hi = brear+berr
            rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
            bx_lo, bx_hi, by_lo, by_hi = bx-rb, bx+rb, by1-rb, by1+rb
            if stack is None:
                root = self._root
                if (root is None or root.min_slot >= probe_end or
                        not self._node_may_cover(root, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom)):
                    checked += 1
                    self._cover_probe = None
                    continue
                if root.idxs is not None:
                    stack = [[root, None]]
                else:
                    checked += 1
                    stack = [[c, None] for c in root.children
                             if c.min_slot < probe_end and
                             self._node_may_cover(c, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom)]
                    if not stack:
                        self._cover_probe = None
                        continue
                self._cover_probe = (cand, stack, probe_end)
                if checked >= budget:
                    break
            found = -1
            while stack and checked < budget:
                frame = stack[-1]
                node = frame[0]
                if type(node) is _RNode:
                    if (node.min_slot >= probe_end or
                            not self._node_may_cover(node, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom)):
                        stack.pop()
                        checked += 1
                        continue
                    if node.idxs is None:
                        stack.pop()
                        checked += 1
                        for c in node.children:
                            if (c.min_slot < probe_end and
                                    self._node_may_cover(c, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom)):
                                stack.append([c, None])
                        continue
                    idxs = node.idxs
                    idx = 0 if frame[1] is None else frame[1]
                    n_idx = len(idxs)
                    if idx >= n_idx:
                        stack.pop()
                        checked += 1
                        continue
                    while idx < n_idx and checked < budget:
                        s = idxs[idx]
                        idx += 1
                        checked += 1
                        if s < probe_end and s != cand and alive[s]:
                            if (hs[s] <= bh if std_geom else hs[s] < bh):
                                idx = n_idx
                                break
                            if _dominates_coords(xs[s], y1s[s], y2s[s], hs[s], rears[s], errs[s],
                                                 bx, by1, by2, bh, brear, berr,
                                                 rlos[s], brlo, rlo2s[s], brlo2):
                                found = s
                                break
                    frame[1] = idx
                    if found >= 0:
                        break
                    if idx >= n_idx:
                        stack.pop()
                else:
                    idx = frame[1]
                    n_idx = len(node)
                    if idx >= n_idx:
                        stack.pop()
                        checked += 1
                        continue
                    while idx < n_idx and checked < budget:
                        s = node[idx]
                        idx += 1
                        checked += 1
                        if s < probe_end and s != cand and alive[s]:
                            if _dominates_coords(xs[s], y1s[s], y2s[s], hs[s], rears[s], errs[s],
                                                 bx, by1, by2, bh, brear, berr,
                                                 rlos[s], brlo, rlo2s[s], brlo2):
                                found = s
                                break
                    frame[1] = idx
                    if found >= 0:
                        break
                    if idx >= n_idx:
                        stack.pop()
            if found >= 0:
                self._kill_slot(cand)
                self._cover_probe = None
                self._cache_dom(found)
                ah = hs[found]; ra = OCCLUSION_TOP_DRIFT_HEIGHTS*ah
                if self._cover_count > 0 and self.count > 1 and self._root_may_be_dominated(
                        xs[found], y1s[found], ah, rears[found]-errs[found],
                        xs[found]-ra, xs[found]+ra, y1s[found]-ra, y1s[found]+ra, std_geom):
                    self._queue_dom_sweep(found, xs[found], y1s[found], y2s[found], ah,
                                          rears[found], errs[found], rlos[found], rlo2s[found])
            elif not stack:
                self._cover_probe = None
                if checked == iter_start:
                    checked += 1
        return checked
    def step_dom_sweep(self, budget):
        if self._own_retired and self._retired:
            _step_retire(self._retired, 64)
        if budget <= 0:
            return 0
        has_ev = self._dom_sweep is not None or bool(self._dom_queue)
        has_cov = self._cover_probe is not None or self._cover_count > 0
        if not has_ev and not has_cov:
            return 0
        if not has_cov:
            return self._step_evict_sweep(budget)
        if not has_ev:
            used = self._step_cover_probe(budget)
            if used < budget and (self._dom_sweep is not None or bool(self._dom_queue)):
                used += self._step_evict_sweep(budget-used)
            return used
        if budget == 1:
            self._sweep_turn = not self._sweep_turn
            return self._step_cover_probe(1) if self._sweep_turn else self._step_evict_sweep(1)
        half = budget//2
        used = self._step_evict_sweep(budget-half)
        used += self._step_cover_probe(budget-used)
        if used < budget and (self._dom_sweep is not None or bool(self._dom_queue)):
            used += self._step_evict_sweep(budget-used)
        return used

    def needs_compact(self):
        return (self._rebuild is not None or
                (self.count == 0 and len(self._x) > 0) or
                len(self._x) > 2*max(self.count, 64) or
                (not self.has_dom_sweep() and len(self._x) > max(8, 8*self.count)))

    def maybe_compact(self, budget=64):
        if self._own_retired and self._retired:
            _step_retire(self._retired, 64)
        if budget <= 0:
            return 0
        if self.count == 0:
            if len(self._x):
                self.clear()
            return 0
        if self._rebuild is None:
            if not (len(self._x) > 2*max(self.count, 64) or
                    (not self.has_dom_sweep() and len(self._x) > max(8, 8*self.count))):
                return 0
            cstack = [self._hroot] if self._hroot is not None else []
            self._rebuild = [_ClipBaselines(exact_rear=self._exact_rear, retire=self._retired),
                             cstack, len(self._x), {}, -1, None]
        return self._step_compact(budget)

    def prune(self, px, y1, *, compact=True):
        if self.count == 0:
            return False
        neg_px, neg_y1 = -px, -y1
        hroot = self._hroot
        if (hroot is None or hroot.max_x_hi < px or hroot.max_neg_x_lo < neg_px or
                hroot.max_y_hi < y1 or hroot.max_neg_y_lo < neg_y1):
            self.clear()
            return False
        x_hi, x_lo, y_hi, y_lo = self._x_hi, self._x_lo, self._y_hi, self._y_lo
        alive = self._alive; blk_by_id = self._blk_by_id; slot_bid = self._slot_bid
        dp = self._dom_pending; cp = self._cover_pending
        xs = self._x; y1s = self._y1; hs = self._h
        rears = self._rear; errs = self._err; rlos = self._rlo; rlo2s = self._rlo2; hpos = self._hpos
        dirty = []
        def flush_dirty():
            parents = set()
            for blk in dirty:
                if blk.refresh_bounds(xs, y1s, hs, propagate=False) and blk.parent is not None:
                    parents.add(blk.parent)
            dirty.clear()
            if parents:
                _h_refresh_ancestors(parents)
        killed, ck = x_hi.pop_below(px, alive, blk_by_id, slot_bid, dp, cp, rears, errs, rlos, rlo2s, hpos, dirty)
        if killed:
            self.count -= killed; self._cover_count -= ck
            if self.count == 0:
                self.clear()
                return False
            if dirty:
                flush_dirty()
                if hroot.max_neg_x_lo < neg_px or hroot.max_y_hi < y1 or hroot.max_neg_y_lo < neg_y1:
                    self.clear()
                    return False
        k2, ck2 = x_lo.pop_below(neg_px, alive, blk_by_id, slot_bid, dp, cp, rears, errs, rlos, rlo2s, hpos, dirty)
        if k2:
            killed += k2; self.count -= k2; self._cover_count -= ck2
            if self.count == 0:
                self.clear()
                return False
            if dirty:
                flush_dirty()
                if hroot.max_y_hi < y1 or hroot.max_neg_y_lo < neg_y1:
                    self.clear()
                    return False
        k3, ck3 = y_hi.pop_below(y1, alive, blk_by_id, slot_bid, dp, cp, rears, errs, rlos, rlo2s, hpos, dirty)
        if k3:
            killed += k3; self.count -= k3; self._cover_count -= ck3
            if self.count == 0:
                self.clear()
                return False
            if dirty:
                flush_dirty()
                if hroot.max_neg_y_lo < neg_y1:
                    self.clear()
                    return False
        k4, ck4 = y_lo.pop_below(neg_y1, alive, blk_by_id, slot_bid, dp, cp, rears, errs, rlos, rlo2s, hpos, dirty)
        if k4:
            killed += k4; self.count -= k4; self._cover_count -= ck4
            if self.count == 0:
                self.clear()
                return False
            if dirty:
                flush_dirty()
        if self._rebuild is not None:
            shadow = self._rebuild[0]
            mapping = self._rebuild[3]
            q_shadow = self._rebuild[5]
            if not shadow.prune(px, y1, compact=False):
                self._rebuild = None
                extra = []
                if q_shadow is not None:
                    q_shadow.retire_to(extra)
                if len(mapping) > 64:
                    extra.append((3, mapping))
                if extra:
                    self._retired.append(extra)
        return True

    def _is_covered(self, node, bx, by1, by2, bh, brear, berr, rb, bx_lo, bx_hi, by_lo, by_hi, brear_hi,
                    budget, std_geom=True, queue_dom=True, brlo=0.0, brlo2=0.0):
        if (node is None or budget[0] <= 0 or (node.max_h <= bh if std_geom else node.max_h < bh) or
                node.max_rear_lo < brear_hi or
                node.min_x_lo > bx_lo or node.max_x_hi < bx_hi or
                node.min_y_lo > by_lo or node.max_y_hi < by_hi):
            return False
        reach = (OCCLUSION_TOP_DRIFT_HEIGHTS*(node.max_h-bh)+
                 max(min(1e-9, 1e-6*node.max_h), 16.0*math.ulp(max(abs(bx), abs(by1), node.max_h))))
        if bx+reach < node.min_x or bx-reach > node.max_x or by1+reach < node.min_y or by1-reach > node.max_y:
            return False
        if node.idxs is not None:
            alive = self._alive
            xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
            rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
            for i in node.idxs:
                budget[0] -= 1
                if alive[i]:
                    if (hs[i] <= bh if std_geom else hs[i] < bh):
                        return False
                    if _dominates_coords(xs[i], y1s[i], y2s[i], hs[i], rears[i], errs[i],
                                         bx, by1, by2, bh, brear, berr, rlos[i], brlo, rlo2s[i], brlo2):
                        was_hot = i in self._dom_cache
                        self._cache_dom(i)
                        if queue_dom and not was_hot and self.count > 1:
                            ah = hs[i]; ra = OCCLUSION_TOP_DRIFT_HEIGHTS*ah
                            if self._root_may_be_dominated(xs[i], y1s[i], ah, rears[i]-errs[i],
                                                           xs[i]-ra, xs[i]+ra, y1s[i]-ra, y1s[i]+ra, self._std_geom):
                                self._queue_dom_sweep(i, xs[i], y1s[i], y2s[i], ah,
                                                      rears[i], errs[i], rlos[i], rlo2s[i])
                        return True
                if budget[0] <= 0:
                    return False
            return False
        budget[0] -= 1
        if budget[0] <= 0:
            return False
        for child in node.children:
            if self._is_covered(child, bx, by1, by2, bh, brear, berr, rb, bx_lo, bx_hi, by_lo, by_hi, brear_hi,
                                budget, std_geom, queue_dom, brlo, brlo2):
                return True
            if budget[0] <= 0:
                return False
        return False

    def covered(self, b, dom_budget=None):
        nb = len(b)
        if nb == 8:
            bx, by1, by2, bh, brear, berr, brlo, brlo2 = b
        elif nb == 7:
            bx, by1, by2, bh, brear, berr, brlo = b
            brlo2 = 0.0
        else:
            bx, by1, by2, bh, brear, berr = b
            brlo = brlo2 = 0.0
        if self._exact_rear:
            brear, brlo, brlo2 = _exact_rear(by2, bh)
            berr = 0.0
        if self._has_exact(bx, by1, by2, bh) or self._covered_by_hot(bx, by1, by2, bh, brear, berr, brlo, brlo2):
            return True
        if dom_budget is None:
            dom_budget = [512]
        if dom_budget[0] <= 0:
            return False
        rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        std_q = (self._std_geom and by1 == by2-bh and berr == 0.0 if self._exact_rear
                 else self._std_geom and by1 == by2-bh and brear == by2-OCCLUSION_BOTTOM_SHARE*bh and berr == _rear_err(by2, bh))
        return self._is_covered(self._root, bx, by1, by2, bh, brear, berr,
                                rb, bx-rb, bx+rb, by1-rb, by1+rb, brear+berr, dom_budget, std_q,
                                brlo=brlo, brlo2=brlo2)

    def _evict_dominated(self, node, ax, ay1, ay2, ah, arear, aerr, ax_lo, ax_hi, ay_lo, ay_hi, arear_lo,
                         budget, skip_slot=-1, std_geom=True, arlo=0.0, arlo2=0.0):
        # Standard geometry guarantees any b != a dominated by a has b.h < ah; synthetic keeps b.h <= ah.
        if (node is None or budget[0] <= 0 or (node.min_h >= ah if std_geom else node.min_h > ah) or
                node.min_rear_hi > arear_lo or
                node.max_x_lo < ax_lo or node.min_x_hi > ax_hi or
                node.max_y_lo < ay_lo or node.min_y_hi > ay_hi):
            return node, False, False
        reach = (OCCLUSION_TOP_DRIFT_HEIGHTS*(ah-node.min_h)+
                 max(min(1e-9, 1e-6*ah), 16.0*math.ulp(max(abs(ax), abs(ay1), ah))))
        if ax+reach < node.min_x or ax-reach > node.max_x or ay1+reach < node.min_y or ay1-reach > node.max_y:
            return node, False, False
        if node.idxs is not None:
            alive = self._alive
            xs, y1s, y2s, hs = self._x, self._y1, self._y2, self._h
            rears, errs, rlos, rlo2s = self._rear, self._err, self._rlo, self._rlo2
            kept = array("I")
            removed = False
            evicted = False
            idx_list = node.idxs
            n_idx = len(idx_list)
            j = 0
            while j < n_idx:
                i = idx_list[j]
                if not alive[i]:
                    removed = True
                    j += 1
                    continue
                if budget[0] <= 0:
                    kept.extend(idx_list[j:])
                    break
                budget[0] -= 1
                if i != skip_slot and _dominates_coords(ax, ay1, ay2, ah, arear, aerr,
                                                        xs[i], y1s[i], y2s[i], hs[i], rears[i], errs[i],
                                                        arlo, rlos[i], arlo2, rlo2s[i]):
                    self._kill_slot(i)
                    removed = True
                    evicted = True
                else:
                    kept.append(i)
                j += 1
            if not removed:
                return node, False, False
            if not kept:
                empty = array("I")
                node.idxs = empty
                node.refresh_leaf(xs, y1s, hs, rears, errs)
                self._on_leaf_compact(node, idx_list, empty)
                return None, True, evicted
            node.idxs = kept
            node.refresh_leaf(xs, y1s, hs, rears, errs)
            self._on_leaf_compact(node, idx_list, kept)
            return node, True, evicted
        budget[0] -= 1
        kept_children = []
        changed = False
        evicted = False
        ch = node.children
        for idx_c, c in enumerate(ch):
            if budget[0] <= 0:
                kept_children.extend(ch[idx_c:])
                break
            nc, chg, ev = self._evict_dominated(c, ax, ay1, ay2, ah, arear, aerr,
                                                ax_lo, ax_hi, ay_lo, ay_hi, arear_lo, budget,
                                                skip_slot=skip_slot, std_geom=std_geom, arlo=arlo, arlo2=arlo2)
            if chg:
                changed = True
            if ev:
                evicted = True
            if nc is not None:
                kept_children.append(nc)
        if not changed:
            return node, False, evicted
        if not kept_children:
            return None, True, evicted
        node.children = kept_children
        node.refresh_internal()
        return node, True, evicted

    def _rtree_insert(self, idx, bx, by1, bh, brear, berr):
        if self._root is None:
            node = _RNode(idxs=array("I", (idx,)))
            node.include_point(idx, bx, by1, bh, brear, berr)
            self._root = node
            return
        path = []
        node = self._root
        while node.idxs is None:
            node.include_point(idx, bx, by1, bh, brear, berr)
            best_c = None
            best_enl = math.inf
            best_perim = math.inf
            best_cnt = 0
            for c in node.children:
                perim = (c.max_x-c.min_x)+(c.max_y-c.min_y)
                nx0 = bx if bx < c.min_x else c.min_x
                nx1 = bx if bx > c.max_x else c.max_x
                ny0 = by1 if by1 < c.min_y else c.min_y
                ny1 = by1 if by1 > c.max_y else c.max_y
                enl = (nx1-nx0)+(ny1-ny0)-perim
                if (enl < best_enl or
                        (enl == best_enl and (perim < best_perim or (perim == best_perim and c.count < best_cnt)))):
                    best_c = c; best_enl = enl; best_perim = perim; best_cnt = c.count
            path.append(node)
            node = best_c
        hs = self._h
        idxs = node.idxs
        k = len(idxs)
        while k > 0 and hs[idxs[k-1]] < bh:
            k -= 1
        idxs.insert(k, idx)
        self._on_leaf_insert(node, k)
        node.include_point(idx, bx, by1, bh, brear, berr)
        if len(idxs) <= _R_LEAF_MAX:
            return
        alive = self._alive
        xs, y1s, rears, errs = self._x, self._y1, self._rear, self._err
        live_leaf = [i for i in idxs if alive[i]]
        if len(live_leaf) <= _R_LEAF_MAX:
            new_idxs = array("I", live_leaf)
            node.idxs = new_idxs
            node.refresh_leaf(xs, y1s, hs, rears, errs)
            self._on_leaf_compact(node, idxs, new_idxs)
            for anc in reversed(path):
                anc.refresh_internal()
            return
        if (node.max_x-node.min_x) >= (node.max_y-node.min_y):
            live_leaf.sort(key=lambda i: (xs[i], y1s[i]))
        else:
            live_leaf.sort(key=lambda i: (y1s[i], xs[i]))
        mid = len(live_leaf)//2
        left_ids = live_leaf[:mid]; left_ids.sort(key=lambda i: -hs[i])
        right_ids = live_leaf[mid:]; right_ids.sort(key=lambda i: -hs[i])
        node.idxs = array("I", left_ids)
        node.refresh_leaf(xs, y1s, hs, rears, errs)
        sibling = _RNode(idxs=array("I", right_ids))
        sibling.refresh_leaf(xs, y1s, hs, rears, errs)
        self._on_leaf_split(node, sibling, idxs)
        while path:
            parent = path.pop()
            parent.children.append(sibling)
            if len(parent.children) <= _R_NODE_MAX:
                return
            ch = parent.children
            if (parent.max_x-parent.min_x) >= (parent.max_y-parent.min_y):
                ch.sort(key=lambda c: (c.min_x+c.max_x, c.min_y+c.max_y))
            else:
                ch.sort(key=lambda c: (c.min_y+c.max_y, c.min_x+c.max_x))
            cmid = len(ch)//2
            parent.children = ch[:cmid]
            parent.refresh_internal()
            sibling = _RNode(children=ch[cmid:])
            sibling.refresh_internal()
            self._on_internal_split(parent, sibling)
        new_root = _RNode(children=[self._root, sibling])
        new_root.refresh_internal()
        self._root = new_root

    @staticmethod
    def _node_may_be_dominated(node, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom=True):
        if (node is None or (node.min_h >= ah if std_geom else node.min_h > ah) or node.min_rear_hi > arear_lo or
                node.max_x_lo < ax_lo or node.min_x_hi > ax_hi or
                node.max_y_lo < ay_lo or node.min_y_hi > ay_hi):
            return False
        reach = (OCCLUSION_TOP_DRIFT_HEIGHTS*(ah-node.min_h)+
                 max(min(1e-9, 1e-6*ah), 16.0*math.ulp(max(abs(ax), abs(ay1), ah))))
        return not (ax+reach < node.min_x or ax-reach > node.max_x or ay1+reach < node.min_y or ay1-reach > node.max_y)

    def _root_may_be_dominated(self, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom=True):
        node = self._root
        if not self._node_may_be_dominated(node, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom):
            return False
        if node.idxs is not None:
            return True
        for c in node.children:
            if self._node_may_be_dominated(c, ax, ay1, ah, arear_lo, ax_lo, ax_hi, ay_lo, ay_hi, std_geom):
                return True
        return False

    @staticmethod
    def _node_may_cover(node, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom=True):
        if (node is None or (node.max_h <= bh if std_geom else node.max_h < bh) or node.max_rear_lo < brear_hi or
                node.min_x_lo > bx_lo or node.max_x_hi < bx_hi or
                node.min_y_lo > by_lo or node.max_y_hi < by_hi):
            return False
        reach = (OCCLUSION_TOP_DRIFT_HEIGHTS*(node.max_h-bh)+
                 max(min(1e-9, 1e-6*node.max_h), 16.0*math.ulp(max(abs(bx), abs(by1), node.max_h))))
        return not (bx+reach < node.min_x or bx-reach > node.max_x or by1+reach < node.min_y or by1-reach > node.max_y)

    def _root_may_cover(self, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom=True):
        node = self._root
        if not self._node_may_cover(node, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom):
            return False
        if node.idxs is not None:
            return True
        for c in node.children:
            if self._node_may_cover(c, bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_geom):
                return True
        return False

    def add(self, b, *, compact=True, dom_budget=None):
        if self._own_retired and self._retired:
            _step_retire(self._retired, 64)
        nb = len(b)
        if nb == 8:
            bx, by1, by2, bh, brear, berr, brlo, brlo2 = b
        elif nb == 7:
            bx, by1, by2, bh, brear, berr, brlo = b
            brlo2 = 0.0
        else:
            bx, by1, by2, bh, brear, berr = b
            brlo = brlo2 = 0.0
        if self._exact_rear:
            brear, brlo, brlo2 = _exact_rear(by2, bh)
            berr = 0.0
        if self._has_exact(bx, by1, by2, bh) or self._covered_by_hot(bx, by1, by2, bh, brear, berr, brlo, brlo2):
            if compact:
                self.maybe_compact()
            return
        if dom_budget is None:
            dom_budget = [512]
        rb = OCCLUSION_TOP_DRIFT_HEIGHTS*bh
        bx_lo, bx_hi, by_lo, by_hi = bx-rb, bx+rb, by1-rb, by1+rb
        brear_hi = brear+berr
        std_q = (self._std_geom and by1 == by2-bh and berr == 0.0 if self._exact_rear
                 else self._std_geom and by1 == by2-bh and brear == by2-OCCLUSION_BOTTOM_SHARE*bh and berr == _rear_err(by2, bh))
        may_be_covered = self._root_may_cover(bx, by1, bh, brear_hi, bx_lo, bx_hi, by_lo, by_hi, std_q)
        if may_be_covered and dom_budget[0] > 0 and self._is_covered(
                self._root, bx, by1, by2, bh, brear, berr, rb, bx_lo, bx_hi, by_lo, by_hi, brear_hi, dom_budget,
                std_q, queue_dom=(self._cover_count > 0 or self._cover_probe is not None), brlo=brlo, brlo2=brlo2):
            if compact:
                self.maybe_compact()
            return
        cover_exhausted = may_be_covered and dom_budget[0] <= 0
        could_evict = self._root_may_be_dominated(bx, by1, bh, brear-berr, bx_lo, bx_hi, by_lo, by_hi, std_q)
        if could_evict and dom_budget[0] > 0:
            self._root, _, _ = self._evict_dominated(self._root, bx, by1, by2, bh, brear, berr,
                                                     bx_lo, bx_hi, by_lo, by_hi, brear-berr, dom_budget,
                                                     std_geom=std_q, arlo=brlo, arlo2=brlo2)
            while self._root is not None and self._root.idxs is None and len(self._root.children) == 1:
                self._root = self._root.children[0]
            if self.count == 0:
                self.clear()
        if not std_q:
            self._std_geom = False
        slot = len(self._x)
        self._x.append(bx); self._y1.append(by1); self._y2.append(by2)
        self._h.append(bh); self._rear.append(brear); self._err.append(berr)
        self._rlo.append(brlo); self._rlo2.append(brlo2)
        self._alive.append(1)
        self._dom_pending.append(0)
        self._cover_pending.append(0)
        self.count += 1
        self._x_hi.push(bx_hi, slot); self._x_lo.push(-bx_lo, slot)
        self._y_hi.push(by_hi, slot); self._y_lo.push(-by_lo, slot)
        self._blocks_add(bh, by2, bx, by1, brear, berr, slot)
        self._rtree_insert(slot, bx, by1, bh, brear, berr)
        if self._rebuild is not None:
            shadow, _, _, mapping, _, _ = self._rebuild
            mapping[slot] = shadow._insert_raw(bx, by1, by2, bh, brear, berr, brlo, brlo2)
        if cover_exhausted and self.count > 1:
            self._queue_cover_probe(slot)
        if could_evict:
            self._cache_dom(slot)
            if self.count > 1 and dom_budget[0] <= 0:
                self._queue_dom_sweep(slot, bx, by1, by2, bh, brear, berr, brlo, brlo2)
        if compact:
            self.maybe_compact()

    def best_for_crossing(self, height_eligible, rear_eligible, rear_floor, height, y1, px, side, best_key):
        """Return highest qualifying live ``(h_base, y2_base)`` beating ``best_key`` (or ``best_key``
        unchanged) via exact right-to-left traversal of the ``(h, y2, x)`` B-tree with live
        exact 3-term rear subtree pruning, straight-line shortcuts, and equal-height skips."""
        root = self._hroot
        if root is None or self.count == 0:
            return best_key
        if isinstance(rear_floor, tuple):
            if len(rear_floor) == 4:
                rf_hi, rf_lo, rf_lo2, rf_lo3 = rear_floor
                exact_q = self._exact_rear
            else:
                rf_hi, rf_lo = rear_floor
                rf_lo2 = rf_lo3 = -math.inf
                exact_q = False
        else:
            rf_hi = rear_floor
            rf_lo = rf_lo2 = rf_lo3 = -math.inf
            exact_q = False
        if (root.max_rear_hi < rf_hi or
                (root.max_rear_hi == rf_hi and
                 (root.max_rear_lo < rf_lo or
                  (root.max_rear_lo == rf_lo and
                   (root.max_rear_lo2 < rf_lo2 or (root.max_rear_lo2 == rf_lo2 and rf_lo3 > 0.0)))))):
            return best_key
        alive = self._alive
        rears = self._rear; errs = self._err; rlos = self._rlo; rlo2s = self._rlo2
        std_geom = self._std_geom
        skip_h = math.inf
        stack = [root]
        while stack:
            node = stack.pop()
            if (node.max_rear_hi < rf_hi or
                    (node.max_rear_hi == rf_hi and
                     (node.max_rear_lo < rf_lo or
                      (node.max_rear_lo == rf_lo and
                       (node.max_rear_lo2 < rf_lo2 or (node.max_rear_lo2 == rf_lo2 and rf_lo3 > 0.0)))))):
                continue
            if type(node) is _HNode:
                if node.min_h >= skip_h:
                    continue
                if not height_eligible(node.max_h):
                    return best_key
                if best_key is not None and (node.max_h, node.max_y2) <= best_key:
                    return best_key
                for c in node.children:
                    if (c.max_rear_hi < rf_hi or
                            (c.max_rear_hi == rf_hi and
                             (c.max_rear_lo < rf_lo or
                              (c.max_rear_lo == rf_lo and
                               (c.max_rear_lo2 < rf_lo2 or (c.max_rear_lo2 == rf_lo2 and rf_lo3 > 0.0)))))):
                        continue
                    c_min_h = c.h[0] if type(c) is _HBlock else c.min_h
                    if c_min_h >= skip_h:
                        continue
                    c_max_h = c.h[-1] if type(c) is _HBlock else c.max_h
                    if not height_eligible(c_max_h):
                        continue
                    c_max_y2 = c.y2[-1] if type(c) is _HBlock else c.max_y2
                    if best_key is not None and (c_max_h, c_max_y2) <= best_key:
                        continue
                    stack.append(c)
                continue
            bh = node.h; by2 = node.y2
            if bh[0] >= skip_h:
                continue
            last_i = len(bh)-1 if bh[-1] < skip_h else bisect.bisect_left(bh, skip_h)-1
            if last_i < 0:
                continue
            top_h = bh[last_i]
            if not height_eligible(top_h):
                return best_key
            if best_key is not None and (top_h, by2[last_i]) <= best_key:
                return best_key
            if not exact_q and bh[0] == top_h and not rear_eligible(by2[last_i], top_h):
                skip_h = top_h
                continue
            top_y2 = y1+top_h
            top_behind = side((px, top_y2)) < 0 if (top_h >= height and math.isfinite(top_y2)) else None
            if top_behind is False and not height_eligible(height):
                return best_key
            bslots = node.slots
            i = last_i
            while i >= 0:
                s = bslots[i]
                if not alive[s]:
                    i -= 1
                    continue
                h_base = bh[i]
                if not height_eligible(h_base):
                    return best_key
                y2_base = by2[i]
                if best_key is not None and (h_base, y2_base) <= best_key:
                    return best_key
                r_hi = rears[s]+errs[s]; r_lo = rlos[s]; r_lo2 = rlo2s[s]
                if (r_hi < rf_hi or
                        (r_hi == rf_hi and
                         (r_lo < rf_lo or
                          (r_lo == rf_lo and (r_lo2 < rf_lo2 or (r_lo2 == rf_lo2 and rf_lo3 > 0.0))))) or
                        (not exact_q and not rear_eligible(y2_base, h_base))):
                    if std_geom or not rear_eligible(y2_base, h_base):
                        skip_h = h_base
                        i = bisect.bisect_left(bh, h_base)-1
                    else:
                        i -= 1
                    continue
                if i == last_i and top_behind is not None:
                    behind = top_behind
                else:
                    unclipped_y2 = y1+h_base
                    if not math.isfinite(unclipped_y2):
                        i -= 1
                        continue
                    behind = side((px, unclipped_y2)) < 0
                if behind:
                    return (h_base, y2_base)
                if h_base >= height:
                    return best_key
                i -= 1
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
        self._maint_cursor = 0
        self._dom_cursor = 0
        self._retired = deque()
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
        y1 = py-height
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(y1) and
                abs(px) <= _MAX_PIXEL_COORD and abs(py) <= _MAX_PIXEL_COORD and abs(y1) <= _MAX_PIXEL_COORD):
            return None
        return px, py, height

    def _step_continuous(self, prev, timestamp, sequence):
        gap_tol = min(1e-9, 1e-6*self.policy.max_gap_sec)
        return bool(prev is not None and not prev.get("handoff") and
                    prev["sequence"] == sequence-1 and
                    0 < timestamp-prev["time"] <= self.policy.max_gap_sec+gap_tol)

    def _clear_clip_state(self, state):
        saved = state.get("clip_baselines")
        if saved is not None:
            saved.retire_to(self._retired)
        state["pending_crossing"] = state["clip_baselines"] = None
        if "clip_history" in state:
            state["clip_history"].clear()

    def _record_history(self, state, prev_obs, observation, timestamp, consecutive):
        history = state.setdefault("history", deque(maxlen=MAX_HISTORY_SAMPLES))
        clip_history = state.setdefault("clip_history", deque(maxlen=MAX_HISTORY_SAMPLES))
        gap_tol = min(1e-9, 1e-6*self.policy.max_gap_sec)
        recent = (prev_obs is not None and not prev_obs.get("handoff") and
                  0 < timestamp-prev_obs["time"] <= self.policy.max_gap_sec+gap_tol)
        height = _height(observation)
        try:px, py = float(observation.point[0]), float(observation.point[1])
        except (TypeError, ValueError, IndexError, OverflowError, KeyError):px = py = float("nan")
        y1 = py-height if height is not None else float("nan")
        valid_sample = (height is not None and math.isfinite(px) and math.isfinite(py) and math.isfinite(y1) and
                        abs(px) <= _MAX_PIXEL_COORD and abs(py) <= _MAX_PIXEL_COORD and abs(y1) <= _MAX_PIXEL_COORD)
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
            r_hi, r_lo, r_lo2 = _exact_rear(y2, h)
            out.append((b[1], y2-h, y2, h, r_hi, 0.0, r_lo, r_lo2))
        return out

    def _prune_clip_baselines(self, state, px, y1):
        saved = state.get("clip_baselines")
        if saved is not None and not saved.prune(px, y1):
            saved.retire_to(self._retired)
            state["clip_baselines"] = None

    def _advance_clip_maintenance(self, maint_states, compact_budget, sweep_budget):
        if not maint_states:
            return
        dom_pending = [s["clip_baselines"] for s in maint_states
                       if not s["row"]["completed"] and s.get("clip_baselines") is not None
                       and s["clip_baselines"].has_dom_sweep()]
        if dom_pending and sweep_budget[0] > 0:
            n = len(dom_pending)
            start = self._maint_cursor % n
            ordered = dom_pending[start:]+dom_pending[:start]
            for idx, saved in enumerate(ordered):
                if sweep_budget[0] <= 0:
                    break
                share = max(1, sweep_budget[0]//(n-idx))
                sweep_budget[0] -= saved.step_dom_sweep(share)
        compact_pending = [s["clip_baselines"] for s in maint_states
                           if not s["row"]["completed"] and s.get("clip_baselines") is not None
                           and s["clip_baselines"].needs_compact()]
        if compact_pending and compact_budget[0] > 0:
            n = len(compact_pending)
            start = self._maint_cursor % n
            ordered = compact_pending[start:]+compact_pending[:start]
            for idx, saved in enumerate(ordered):
                if compact_budget[0] <= 0:
                    break
                share = max(1, compact_budget[0]//(n-idx))
                compact_budget[0] -= saved.maybe_compact(share)
        if dom_pending or compact_pending:
            self._maint_cursor += 1

    def _refresh_clip_baselines(self, state, observation, timestamp, prev_obs, dom_budget):
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
        if saved is not None:
            uncovered = [p for p in pairs if not saved.covered(p, dom_budget)]
            if not uncovered:
                return
        else:
            uncovered = pairs
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
        for b in uncovered:
            r = OCCLUSION_TOP_DRIFT_HEIGHTS*b[3]
            if (y1_span <= r and x_span <= r and
                    top_speed < OCCLUSION_STATIC_TOP_SPEED*b[3] and
                    b[0]-r <= px <= b[0]+r and b[1]-r <= y1 <= b[1]+r):
                valid.append(b)
        if valid:
            if saved is None:
                saved = _ClipBaselines(exact_rear=True, retire=self._retired)
                state["clip_baselines"] = saved
            for b in valid:
                saved.add(b, compact=False, dom_budget=dom_budget)

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
        shrink_tol = min(1e-9, 1e-6*height)
        rear_tol = min(1e-9, 1e-6*height)
        def height_eligible(h_base):
            return (height <= CROSSING_CLIP_SHRINK_RATIO*h_base+shrink_tol and
                    y1_span <= OCCLUSION_TOP_DRIFT_HEIGHTS*h_base and
                    x_span <= OCCLUSION_TOP_DRIFT_HEIGHTS*h_base and
                    top_speed < OCCLUSION_STATIC_TOP_SPEED*h_base)
        rear_floor = _exact_floor(py, height, rear_tol)
        def rear_eligible(y2_base, h_base):
            return (*_exact_rear(y2_base, h_base), 0.0) >= rear_floor
        for b in self._corroborated_pairs(earlier):
            y2_base, h_base = b[2], b[3]
            if not (height_eligible(h_base) and (b[4], b[6], b[7], 0.0) >= rear_floor):
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
            saved_key = saved.best_for_crossing(height_eligible, rear_eligible, rear_floor,
                                                height, y1, px, self.side, best_key)
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

    @staticmethod
    def _invalid_crossing_geometry(observation):
        if observation.bbox is not None and _raw_box(observation.bbox) is None:
            return True
        h = _height(observation)
        if observation.height is not None and h is None:
            return True
        try:px, py = float(observation.point[0]), float(observation.point[1])
        except (TypeError, ValueError, IndexError, OverflowError, KeyError):return True
        if not (math.isfinite(px) and math.isfinite(py) and
                abs(px) <= _MAX_PIXEL_COORD and abs(py) <= _MAX_PIXEL_COORD):
            return True
        if h is None:
            return False
        y1 = py-h
        return not (math.isfinite(y1) and abs(y1) <= _MAX_PIXEL_COORD)

    def _evaluate_crossing(self, state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                           valid_motion, continuing, keep_pending, dom_budget):
        row = state["row"]
        pending = state.get("pending_crossing")
        geom = self._valid_box_geometry(observation) if (continuing or pending is not None) else None
        if continuing:
            if self._invalid_crossing_geometry(observation) or (geom is None and pending is not None):
                self._clear_clip_state(state)
                return False
            if geom is None:
                self._clear_clip_state(state)
            else:
                self._prune_clip_baselines(state, geom[0], geom[1]-geom[2])
        if prev_obs.get("crossing_broken"):
            return False
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
                clipped = self._bottom_clipped_crossing(state, observation, timestamp, prev_obs) if continuing else None
                if clipped is not None:
                    h_base, pair = clipped
                    if pair is not None:
                        saved = state.get("clip_baselines")
                        if saved is None:
                            saved = _ClipBaselines(exact_rear=True, retire=self._retired)
                            state["clip_baselines"] = saved
                        saved.add(pair, compact=False, dom_budget=dom_budget)
                    if keep_pending and h_base > pending["h_base"]:
                        pending["h_base"] = h_base
                    return False
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
                        saved = _ClipBaselines(exact_rear=True, retire=self._retired)
                        state["clip_baselines"] = saved
                    saved.add(pair, compact=False, dom_budget=dom_budget)
                if keep_pending:
                    state["pending_crossing"] = {"from": prev_obs, "h_base": h_base, "recovered": 0}
                return False
            self._cross(row, prev_obs, observation, phase, timestamp, cycle_id)
            self._clear_clip_state(state)
            return True
        if continuing:
            self._refresh_clip_baselines(state, observation, timestamp, prev_obs, dom_budget)
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
        assoc_limit = self.policy.association_gap_sec+(min(1e-9, 1e-6*self.policy.association_gap_sec)
                                                       if (continuing or native_exit) else 0.0)
        if continuing:
            # Read more than once (prepasses, accounting): a one-shot iterator is materialized once.
            observations = observations if isinstance(observations, (list, tuple)) else list(observations)
            # A sighting older than the association timeout no longer makes an ID "not newborn".
            if self.outside_seen:
                self.outside_seen = {k: t for k, t in self.outside_seen.items()
                                     if timestamp-t <= assoc_limit}
            # A handed-off ID seen again contests the live link before any row of this
            # frame completes (e.g. the continuing ID crossing the line): order-independent.
            if self.handed_off:
                for observation in observations:
                    if observation.local_id is None or not observation.stable_id:continue
                    if native_exit and self._invalid_crossing_geometry(observation):continue
                    identity = (observation.epoch, observation.local_id)
                    if identity in self.handed_off:
                        self._contest(self.handed_off[identity]["row_id"], identity, timestamp)
            self._continue_fragments(observations, timestamp, unassigned)
        present = set()
        compact_budget = [64]
        dom_budget = [384]
        sweep_budget = [256]
        maint_states = []
        n_obs = len(observations) if continuing else 0
        dom_window = 24
        dom_start = (self._dom_cursor*dom_window) % n_obs if n_obs > dom_window else 0
        if n_obs > dom_window:
            self._dom_cursor += 1
        dom_rem = min(n_obs, dom_window)
        obs_seen = 0
        for obs_idx, observation in enumerate(observations):
            obs_seen = obs_idx+1
            in_dom = (n_obs <= dom_window) or ((obs_idx-dom_start) % n_obs < dom_window)
            obs_dom_rem = dom_rem if in_dom else 0
            if in_dom and dom_rem > 0:
                dom_rem -= 1
            if observation.local_id is None or not observation.stable_id or not _finite_point(observation.point): continue
            if native_exit and self._invalid_crossing_geometry(observation): continue
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
                    step_continuous = (self._step_continuous(prev_obs, timestamp, sequence) if (continuing or native_exit) else
                        bool(previous is not None and not previous.get("handoff") and
                             previous["sequence"] == sequence-1 and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec))
                    crossed_now = False
                    geom = self._valid_box_geometry(observation) if continuing else None
                    crossing_broken = bool(continuing and (self._invalid_crossing_geometry(observation) or
                                                           (geom is None and state.get("pending_crossing") is not None)))
                    if step_continuous:
                        share = max(1, dom_budget[0]//obs_dom_rem) if (continuing and obs_dom_rem > 0 and dom_budget[0] > 0) else 0
                        track_dom = [share]
                        crossed_now = self._evaluate_crossing(
                            state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                            False, continuing, keep_pending=native_exit, dom_budget=track_dom)
                        dom_budget[0] -= share-track_dom[0]
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
                            "point": observation.point, "crossing_broken": crossing_broken}
                        if continuing:
                            if geom is not None:
                                state["last_valid_box"] = (timestamp, geom[0], geom[1], geom[2])
                            self._record_history(state, prev_obs, observation, timestamp, step_continuous)
                        state["stop_candidate"] = state["move_candidate"] = None
                        exit_tol = min(1e-8, 1e-6*self.policy.exit_persistence_sec)
                        if timestamp > state["exit_candidate"] and timestamp-state["exit_candidate"] >= self.policy.exit_persistence_sec-exit_tol:
                            row.update(completed=True,completed_at=state["exit_candidate"],tracking_state="EXITED",right_censored=False);self.work+=1
                            self._clear_clip_state(state)
                            self._clear_native_exit(state)
                        else:
                            row["tracking_state"] = "EXIT_PENDING"
                            if continuing and state.get("clip_baselines") is not None:
                                maint_states.append(state)
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
            if continuing or native_exit:
                continuous = not was_exit_pending and self._step_continuous(previous, timestamp, sequence)
                crossing_continuous = self._step_continuous(prev_obs, timestamp, sequence)
            else:
                continuous = (not was_exit_pending and previous is not None and not previous.get("handoff") and
                              previous["sequence"] == sequence-1 and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec)
                crossing_continuous = continuous
            geom = self._valid_box_geometry(observation) if continuing else None
            crossing_broken = bool(continuing and (self._invalid_crossing_geometry(observation) or
                                                   (geom is None and state.get("pending_crossing") is not None)))
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
                share = max(1, dom_budget[0]//obs_dom_rem) if (continuing and obs_dom_rem > 0 and dom_budget[0] > 0) else 0
                track_dom = [share]
                self._evaluate_crossing(
                    state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                    valid_motion, continuing, keep_pending=True, dom_budget=track_dom)
                dom_budget[0] -= share-track_dom[0]
                if continuing and not row["completed"] and state.get("clip_baselines") is not None:
                    maint_states.append(state)
            elif prev_obs is not None or crossing_broken:
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
                if geom is not None:
                    state["last_valid_box"] = (timestamp, geom[0], geom[1], geom[2])
                self._record_history(state, prev_obs, observation, timestamp, crossing_continuous)
            state["previous"] = {"time": timestamp, "sequence": sequence,
                "phase": str(phase), "cycle": cycle_id, "valid_motion": valid_motion,
                "point": observation.point, "crossing_broken": crossing_broken}
        if maint_states:
            self._advance_clip_maintenance(maint_states, compact_budget, sweep_budget)
        for identity, state in list(self.active.items()):
            row = state["row"]
            if row["vehicle_id"] not in self.records and identity not in present and timestamp-state.get("_last_seen",row["last_seen"])>assoc_limit:
                self._clear_clip_state(state)
                del self.active[identity]
                continue
            if identity not in present and not row["completed"]:
                row["tracking_state"] = "LOST"
                self._clear_clip_state(state)
                exit_last = state.get("exit_last") if native_exit else None
                last_observed = max(row["last_seen"], exit_last["time"]) if exit_last is not None else row["last_seen"]
                if timestamp-last_observed > assoc_limit:
                    row.update(completed=True, completed_at=timestamp);self.work+=1
                    self._flag(row, "tracking_lost")
                    self._clear_native_exit(state)
        if self._retired:
            _step_retire(self._retired, max(256, 4*(obs_seen+len(self.active))))

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
        native_exit = self.policy.exit_persistence_sec > 0
        newborns = []
        for o in observations:
            if o.local_id is None or not o.stable_id or not o.in_region or not _finite_point(o.point):continue
            if native_exit and self._invalid_crossing_geometry(o):continue
            identity = (o.epoch, o.local_id)
            height = _height(o)
            if height is None:continue
            y1 = float(o.point[1])-height
            if math.isfinite(y1) and abs(y1) <= _MAX_PIXEL_COORD and self._fresh(identity):
                newborns.append((identity, o, height))
        if not newborns:return
        present = {(o.epoch, o.local_id) for o in observations
                   if o.local_id is not None and o.stable_id and
                   not (native_exit and self._invalid_crossing_geometry(o))}
        lost = []
        # Lost stationary or short-lived vehicles within OCCLUSION_MAX_GAP_SEC cannot
        # link past STATIC_CONTINUATION_MAX_GAP_SEC (or with <3 sightings), but still
        # veto extended occlusion-tail links when any occlusion candidate is present.
        static_vetoes = []
        occlusion_count = 0
        gap_tol = min(1e-9, 1e-6*self.policy.max_gap_sec)
        veto_window = min(self.policy.max_gap_sec, OCCLUSION_MAX_GAP_SEC)+gap_tol
        for identity, state in self.active.items():
            row, previous, history = state["row"], state["previous"], state.get("history")
            if identity in present or row["completed"] or row["vehicle_id"] not in self.records:continue
            if native_exit and state.get("exit_candidate") is not None:
                last_box = state.get("last_valid_box")
                if last_box is not None and 0 < timestamp-last_box[0] <= veto_window:
                    _, bx, by, bh = last_box
                    static_vetoes.append((identity, bx, by, by-bh, bh))
                continue
            if previous is None or not history:continue
            dt = timestamp-previous["time"]
            if not 0 < dt <= self.policy.max_gap_sec+gap_tol:continue
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
        if self._retired:
            _step_retire(self._retired, max(256, 4*len(self.active)))

    def close(self, timestamp, reason="recording_end"):
        self.invalidate(timestamp, reason)
        while self._retired:
            _step_retire(self._retired, 4096)
        for row in self.records.values():
            if not row["completed"]: row.update(completed=True, completed_at=timestamp);self.work+=1

    def rows(self, offset=0, limit=100, include_intervals=False):
        rows = list(self.records.values())
        return [{k: (round(v, 6) if isinstance(v, float) else deepcopy(v))
                 for k, v in row.items() if include_intervals or k != "intervals"}
                for row in rows[offset:offset+limit]]
