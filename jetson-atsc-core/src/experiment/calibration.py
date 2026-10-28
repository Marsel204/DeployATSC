from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.common.source_identity import public_data, public_source, resolve_source
from src.analytics.homography_engine import HomographyEngine
from src.perception.lens_rectifier import LensRectifier

ROOT = Path(__file__).resolve().parents[2]


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Geometry values must be finite")
    return value


def validate_roi(roi, size):
    if not isinstance(roi, dict) or set(roi) != {"xmin", "ymin", "xmax", "ymax"}:
        raise ValueError("A complete signal ROI is required")
    result = {key: int(_finite(value)) for key, value in roi.items()}
    if not (0 <= result["xmin"] < result["xmax"] <= size[0] and
            0 <= result["ymin"] < result["ymax"] <= size[1]):
        raise ValueError("Signal ROI is outside this camera's dimensions or has zero area")
    return result


def legacy_system_payload(config, source, image_size):
    """Read-only migration: a road-source lamp never becomes acquisition ROI."""
    hom = deepcopy(config["homography"])
    return {"role": "system", "source": source, "image_size": image_size,
            "anchors": hom["src_points_pixel"],
            "polygon": hom.get("corridor_polygon_pixel", hom["src_points_pixel"]),
            "road_width_m": hom["ground_dimensions_m"]["width"],
            "road_length_m": hom["ground_dimensions_m"]["length"],
            "rectification_path": config.get("system", {}).get("calibration_path"),
            "lane_metrics": deepcopy(config.get("lane_metrics", {}))}


class CameraCalibration:
    def __init__(self, payload, root=ROOT):
        data = deepcopy(payload)
        self.source = resolve_source(data.get("source", ""))
        data["source"] = public_source(self.source)
        data = public_data(data)
        if data.get("role") not in ("system", "acquisition"):
            raise ValueError("Camera role must be system or acquisition")
        self.role = data["role"]
        if not str(data.get("source", "")).strip():
            raise ValueError("Camera source is required")
        size = data.get("image_size", [])
        if len(size) != 2:
            raise ValueError("Processed image dimensions are required")
        self.size = tuple(int(_finite(x)) for x in size)
        if min(self.size) < 1 or max(self.size) > 8192:
            raise ValueError("Invalid image dimensions")
        self.rectifier = None
        path = data.get("rectification_path")
        if path:
            resolved = Path(path)
            if not resolved.is_absolute(): resolved = Path(root) / resolved
            actual_digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if data.get("rectification_digest") not in (None, actual_digest):
                raise ValueError("Lens content digest changed; review geometry and create a new binding")
            self.rectifier = LensRectifier(str(resolved))
            if self.rectifier.dim != self.size:
                raise ValueError("Lens calibration dimensions differ from this camera")
            data["rectification_digest"] = actual_digest
        elif data.get("rectification_digest"):
            raise ValueError("Lens digest requires its calibration file")

        def points(key, count=None):
            raw = data.get(key)
            if not isinstance(raw, list) or (count is not None and len(raw) != count) or len(raw) < 2:
                raise ValueError(f"Invalid {key}")
            result = []
            for point in raw:
                if len(point) != 2: raise ValueError(f"Invalid {key} point")
                x, y = map(_finite, point)
                if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
                    raise ValueError(f"{key} is outside this camera's dimensions")
                result.append([x, y])
            return result

        anchors = points("anchors", 4)
        if not cv2.isContourConvex(np.array(anchors, np.float32)):
            raise ValueError("Perspective anchors must form an ordered convex quadrilateral")
        if not data.get("polygon"): data["polygon"] = deepcopy(anchors)
        polygon = points("polygon")
        if len(polygon)>3 and polygon[0]==polygon[-1]:
            polygon.pop();data["polygon"]=deepcopy(polygon)
        if len(polygon) < 3 or abs(cv2.contourArea(np.array(polygon, np.float32))) < 1:
            raise ValueError("Waiting/road polygon must have nonzero area")
        def orientation(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        def on_segment(a,b,c):
            return min(a[0],b[0])-1e-8<=c[0]<=max(a[0],b[0])+1e-8 and min(a[1],b[1])-1e-8<=c[1]<=max(a[1],b[1])+1e-8
        for i in range(len(polygon)):
            for j in range(i+1,len(polygon)):
                if (i+1)%len(polygon)==j or (j+1)%len(polygon)==i:continue
                a,b=polygon[i],polygon[(i+1)%len(polygon)];c,d=polygon[j],polygon[(j+1)%len(polygon)]
                o1,o2,o3,o4=orientation(a,b,c),orientation(a,b,d),orientation(c,d,a),orientation(c,d,b)
                crossing=o1*o2<0 and o3*o4<0
                touching=any(abs(o)<1e-8 and on_segment(x,y,z) for o,x,y,z in ((o1,a,b,c),(o2,a,b,d),(o3,c,d,a),(o4,c,d,b)))
                if crossing or touching:raise ValueError("Road/waiting polygon must not self-intersect")
        width = _finite(data.get("road_width_m", 8))
        length = _finite(data.get("road_length_m", 40))
        if width <= 0 or length <= 0: raise ValueError("Ground dimensions must be positive")
        ground = [[0, length], [width, length], [width, 0], [0, 0]]
        self.homography = HomographyEngine(anchors, ground, width, length, polygon)
        if np.linalg.cond(self.homography.H) > 1e7:
            raise ValueError("Perspective geometry is ill-conditioned")
        # Explicit dimension evidence selects the dimension transform; a ruler
        # left in an older draft must not silently rescale that measurement.
        if (data.get("scale_evidence") or {}).get("method") == "measured_dimensions":
            data.pop("ruler", None)
        if data.get("ruler"):
            ruler = data["ruler"]
            ruler_points = {"ruler_points": [ruler["p1"], ruler["p2"]]}
            data.update(ruler_points); points("ruler_points", 2)
            ruler_length = _finite(ruler["length_m"])
            if ruler_length <= 0 or np.linalg.norm(np.subtract(ruler["p1"], ruler["p2"])) < 1:
                raise ValueError("Ruler needs distinct endpoints and positive length")
            self.homography.calibrate_from_sample_ruler(ruler["p1"], ruler["p2"], ruler_length)
            data.pop("ruler_points")
        self.roi = None
        self.lamp_rois = {}
        self.stop_line = None
        self.crossing_direction = int(data.get("crossing_direction", 1))
        if self.role == "acquisition":
            self.roi = validate_roi(data.get("signal_roi"), self.size)
            for phase, region in data.get("lamp_rois", {}).items():
                if phase not in ("RED", "YELLOW", "GREEN"): raise ValueError("Unknown lamp region")
                region = validate_roi(region, self.size)
                if not (self.roi["xmin"] <= region["xmin"] < region["xmax"] <= self.roi["xmax"] and
                        self.roi["ymin"] <= region["ymin"] < region["ymax"] <= self.roi["ymax"]):
                    raise ValueError("Lamp regions must be inside the signal-head box")
                self.lamp_rois[phase] = region
            self.stop_line = points("stop_line", 2)
            if np.linalg.norm(np.subtract(*self.stop_line)) < 1: raise ValueError("Stop line has zero length")
            if self.crossing_direction not in (-1, 1): raise ValueError("Crossing direction must be -1 or 1")
        evidence = data.get("scale_evidence") or {}
        method = evidence.get("method")
        if evidence and (method not in ("measured_dimensions", "ruler") or not str(evidence.get("reference", "")).strip()):
            raise ValueError("Scale evidence needs a measurement method and reference")
        if method == "measured_dimensions" and not all(key in data for key in ("road_width_m", "road_length_m")):
            raise ValueError("Measured dimensions must be explicitly supplied")
        if method == "ruler" and not data.get("ruler"):
            raise ValueError("Ruler scale evidence requires ruler endpoints and length")
        binding = data.get("binding") or {}
        geometry_keys = ("role", "source", "image_size", "rectification_digest", "anchors", "polygon",
                         "road_width_m", "road_length_m", "ruler", "signal_roi", "lamp_rois", "stop_line", "crossing_direction")
        geometry = {key:data.get(key) for key in geometry_keys}
        geometry.update(road_width_m=width, road_length_m=length,
                        lamp_rois=data.get("lamp_rois") or {}, crossing_direction=self.crossing_direction)
        def canonical(value):
            # JSON 8 and 8.0 / API tuple-to-list conversion do not change geometry.
            if isinstance(value, (int,float)) and not isinstance(value,bool):return float(value)
            if isinstance(value, dict):return {key:canonical(item) for key,item in value.items()}
            if isinstance(value, (list,tuple)):return [canonical(item) for item in value]
            return value
        geometry_digest = hashlib.sha256(json.dumps(canonical(geometry),
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        if binding.get("geometry_digest") not in (None, geometry_digest):
            raise ValueError("Geometry binding changed; review and create a new binding")
        if binding.get("reviewed") is True:
            data["binding"] = {"reviewed":True, "geometry_digest":geometry_digest}
        reasons = []
        if not evidence:reasons.append("scale_evidence_missing")
        if binding.get("reviewed") is not True:reasons.append("geometry_binding_unreviewed")
        self.scale_status = {"state":"review_required" if reasons else "verified", "method":method, "reasons":reasons}
        self.payload = data
        self.revision = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]

    def transform(self, frame):
        if tuple(frame.shape[1::-1]) != self.size:
            raise ValueError(f"{self.role} source dimensions changed; recalibration is required")
        return self.rectifier.rectify(frame) if self.rectifier else frame.copy()

    def side(self, point):
        a, b = self.stop_line
        dx, dy = b[0]-a[0], b[1]-a[1]
        return self.crossing_direction * (dx*(point[1]-a[1])-dy*(point[0]-a[0])) / math.hypot(dx, dy)

    def legacy_metrics_config(self):
        h = self.homography
        polygon=np.array([h.pixel_to_ground(*point) for point in self.payload["polygon"]],np.float32)
        area=abs(float(cv2.contourArea(polygon)))
        if not math.isfinite(area) or area<=0:raise ValueError("Road polygon has invalid physical area")
        lane=deepcopy(self.payload.get("lane_metrics",{}))
        lane["physical_road_area_m2"]=area
        return {"homography": {"src_points_pixel": h.src_points.tolist(),
                "dst_points_ground": h.dst_points.tolist(),
                "ground_dimensions_m": {"width": h.ground_width, "length": h.ground_length},
                "corridor_polygon_pixel": h.corridor_poly.tolist()},
                "lane_metrics": lane}
