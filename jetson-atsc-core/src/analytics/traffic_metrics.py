"""
Indonesian PKJI 2014 Traffic Engineering Parameter Extractor.

Extracts:
1. Weighted Vehicle Count (V_w) using Indonesian PKJI 2014 Passenger Car Unit (PCU) factors:
   - Motorcycle (MC): 0.4 PCU
   - Light Vehicle / Car (LV): 1.0 PCU
   - Heavy Vehicle / Bus / Truck (HV): 1.6 PCU
2. Queue Length (Q): Maximum longitudinal metric distance (Y_m) from the stop line
   for all vehicles with ground velocity <= 1.0 m/s.
3. Perspective-Invariant Fixed-Area Lane Occupancy (L):
   L = min(100.0, (2.0*N_MC + 8.0*N_LV + 24.0*N_HV) / A_ROI * 100.0)
   where A_ROI = 472.5 m^2 (10.5m x 45.0m).
4. Kamera Akuisisi Stop-Line Tripwire: Tracks discharged vehicles (N_served) during green phase.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.perception.detector_tracker import TrackedVehicle
from src.analytics.homography_engine import HomographyEngine


class TrafficMetricsExtractor:
    """
    Computes real-time PKJI 2014 traffic parameters and stop-line discharge metrics.
    """

    def __init__(
        self,
        homography: HomographyEngine,
        pcu_weights: Optional[Dict[str, float]] = None,
        footprints_m2: Optional[Dict[str, float]] = None,
        physical_road_area_m2: float = 472.5,
        stopped_speed_threshold_mps: float = 1.0,
        stop_line_tripwire_y: int = 980,
        min_stopped_observations: int = 5,
        strict_motion: bool = False,
    ) -> None:
        self.homography = homography
        self.physical_road_area = physical_road_area_m2
        self.stopped_speed_threshold = stopped_speed_threshold_mps
        self.stop_line_tripwire_y = stop_line_tripwire_y
        self.min_stopped_observations = min_stopped_observations
        self.strict_motion = strict_motion

        # PKJI 2014 PCU weights
        self.pcu_weights = pcu_weights or {
            "motorcycle": 0.4,
            "car": 1.0,
            "truck": 1.6,
            "bus": 1.6,
        }

        # Class footprint areas in m^2
        self.footprints = footprints_m2 or {
            "motorcycle": 2.0,
            "car": 8.0,
            "truck": 24.0,
            "bus": 24.0,
        }

        # Stop-line tripwire tracking for Camera 2: {track_id: last_y}
        self.tripwire_history: Dict[int, float] = {}
        self.served_track_ids: Set[int] = set()
        self.n_served: int = 0

    @classmethod
    def from_config(
        cls,
        config_dict: Dict[str, Any],
        homography: HomographyEngine,
        strict_motion: bool = False,
    ) -> TrafficMetricsExtractor:
        """Instantiates extractor using intersection_roi.json parameters."""
        lane_cfg = config_dict.get("lane_metrics", {})
        return cls(
            homography=homography,
            pcu_weights=lane_cfg.get("pcu_weights"),
            footprints_m2=lane_cfg.get("vehicle_footprints_m2"),
            physical_road_area_m2=float(lane_cfg.get("physical_road_area_m2", 472.5)),
            stopped_speed_threshold_mps=float(lane_cfg.get("stopped_speed_threshold_mps", 1.0)),
            stop_line_tripwire_y=int(lane_cfg.get("stop_line_tripwire_y_pixel", 980)),
            min_stopped_observations=int(lane_cfg.get("min_stopped_observations", 5)),
            strict_motion=strict_motion,
        )

    def extract_approach_metrics(
        self,
        vehicles: List[TrackedVehicle],
    ) -> Dict[str, Any]:
        """
        Extracts PKJI traffic parameters (V_w, Q, L) from approach vehicles.

        Args:
            vehicles: Tracked vehicles detected by Kamera Sistem.

        Returns:
            Dictionary containing V_w, Q, L, vehicle counts, and queue statistics.
        """
        n_mc = 0
        n_lv = 0
        n_hv = 0

        stopped_distances: List[float] = []
        corridor_vehicles: List[TrackedVehicle] = []
        motion_unknown = 0

        for v in vehicles:
            u, v_px = v.bottom_center

            # 1. Filter tracks within the approach corridor polygon
            if not self.homography.is_in_approach_corridor(u, v_px):
                continue

            corridor_vehicles.append(v)
            c_name = v.class_name.lower()

            # Class count accumulation
            if "motorcycle" in c_name or "bike" in c_name:
                n_mc += 1
            elif "truck" in c_name or "bus" in c_name:
                n_hv += 1
            else:
                n_lv += 1

            # 2. Queue accumulation: Stopped vehicle ground distance with temporal persistence gate
            # Projection: Y_m is longitudinal distance from the stop line [0, 45.0m]
            if v.ground_pos is not None:
                _, y_m = v.ground_pos
            else:
                _, y_m = self.homography.pixel_to_ground(u, v_px)

            is_persistent = getattr(v, "obs_count", 5) >= self.min_stopped_observations
            motion_valid = (is_persistent and getattr(v,"stable_id",False) and
                            getattr(v,"velocity_valid",False) and
                            v.velocity_mps is not None and math.isfinite(v.velocity_mps) and
                            v.velocity_mps >= 0 and math.isfinite(y_m))
            if self.strict_motion and not motion_valid:
                motion_unknown += 1
                continue
            if is_persistent and v.velocity_mps <= self.stopped_speed_threshold:
                stopped_distances.append(y_m)

        # 3. Queue Length (Q): Max distance of stopped vehicles from stop line
        Q = max(stopped_distances) if stopped_distances else 0.0
        # Physical bounds clamp [0.0, 45.0m]
        Q = max(0.0, min(self.homography.ground_length, float(Q)))
        complete = self.strict_motion and motion_unknown == 0
        if self.strict_motion and motion_unknown:
            Q = Q if stopped_distances else None
        q_state = ("legacy_unverified" if not self.strict_motion else
                   "partial" if motion_unknown and stopped_distances else
                   "unavailable" if motion_unknown else "measured_zero" if Q == 0 else "measured")
        quality = {
            "V_w":{"available":True,"complete":True,"state":"observed_count"},
            "L":{"available":self.physical_road_area>0,"complete":self.physical_road_area>0,
                 "state":"footprint_estimate"},
            "Q":{"available":Q is not None,"complete":complete,"state":q_state,
                 "observed_vehicle_count":len(corridor_vehicles),
                 "motion_valid_vehicle_count":len(corridor_vehicles)-motion_unknown if self.strict_motion else None,
                 "motion_unknown_vehicle_count":motion_unknown if self.strict_motion else None,
                 "reasons":["motion_unknown"] if motion_unknown else [] if self.strict_motion else ["strict_motion_disabled"]},
            "scale":{"state":"unverified"},
        }

        # 4. Weighted Vehicle Count (V_w): PKJI 2014
        V_w = (
            n_mc * self.pcu_weights.get("motorcycle", 0.4)
            + n_lv * self.pcu_weights.get("car", 1.0)
            + n_hv * self.pcu_weights.get("truck", 1.6)
        )

        # 5. Lane Occupancy Ratio (L): Class footprint substitution over physical area
        occupied_area_m2 = (
            n_mc * self.footprints.get("motorcycle", 2.0)
            + n_lv * self.footprints.get("car", 8.0)
            + n_hv * self.footprints.get("truck", 24.0)
        )
        if self.physical_road_area > 0:
            L = (occupied_area_m2 / self.physical_road_area) * 100.0
        else:
            L = 0.0
        L = min(100.0, max(0.0, float(L)))

        return {
            "V_w": round(float(V_w), 2),
            "Q": round(float(Q), 2) if Q is not None else None,
            "quality": quality,
            "L": round(float(L), 2),
            "n_mc": n_mc,
            "n_lv": n_lv,
            "n_hv": n_hv,
            "total_corridor_vehicles": len(corridor_vehicles),
            "stopped_vehicles_count": len(stopped_distances),
            "occupied_area_m2": round(occupied_area_m2, 2),
            "corridor_vehicles": corridor_vehicles,
        }

    def extract_single_frame_metrics(
        self,
        vehicles: List[TrackedVehicle],
    ) -> Dict[str, Any]:
        """Counts, footprint occupancy and a labelled spatial queue proxy from ONE still image.

        A single image has no motion evidence: speed and stopped state are unknown, so Q is
        the farthest visible corridor vehicle from the stop line (an incomplete diagnostic
        proxy), never a measured stopped queue. Stopped count is unknown (None), not zero.
        """
        n = {"mc":0, "lv":0, "hv":0}
        corridor: List[TrackedVehicle] = []
        extents: List[float] = []
        for v in vehicles:
            u, v_px = v.bottom_center
            if not self.homography.is_in_approach_corridor(u, v_px):
                continue
            corridor.append(v)
            c_name = v.class_name.lower()
            key = "mc" if "motorcycle" in c_name or "bike" in c_name else "hv" if "truck" in c_name or "bus" in c_name else "lv"
            n[key] += 1
            _, y_m = self.homography.pixel_to_ground(u, v_px)
            if math.isfinite(y_m):
                extents.append(y_m)
        q = max(0.0, min(self.homography.ground_length, float(max(extents)))) if extents else 0.0
        V_w = (n["mc"] * self.pcu_weights.get("motorcycle", 0.4) + n["lv"] * self.pcu_weights.get("car", 1.0)
               + n["hv"] * self.pcu_weights.get("truck", 1.6))
        area = (n["mc"] * self.footprints.get("motorcycle", 2.0) + n["lv"] * self.footprints.get("car", 8.0)
                + n["hv"] * self.footprints.get("truck", 24.0))
        L = min(100.0, max(0.0, area / self.physical_road_area * 100.0)) if self.physical_road_area > 0 else 0.0
        reasons = ["single_frame_no_motion_evidence", "stopped_state_unknown"]
        if not corridor:
            reasons.append("no_visible_corridor_vehicles")
        quality = {
            "V_w":{"available":True,"complete":True,"state":"observed_count"},
            "L":{"available":self.physical_road_area>0,"complete":self.physical_road_area>0,
                 "state":"footprint_estimate"},
            "Q":{"available":True, "complete":False, "measured":False,
                 "state":"single_frame_spatial_proxy", "method":"single_frame_spatial_extent",
                 "definition":"farthest visible approach-corridor vehicle from the stop line in one still image",
                 "observed_vehicle_count":len(corridor), "motion_valid_vehicle_count":0,
                 "motion_unknown_vehicle_count":len(corridor), "reasons":reasons},
            "scale":{"state":"unverified"},
        }
        return {"V_w":round(float(V_w), 2), "Q":round(q, 2), "quality":quality, "L":round(float(L), 2),
                "n_mc":n["mc"], "n_lv":n["lv"], "n_hv":n["hv"], "total_corridor_vehicles":len(corridor),
                "stopped_vehicles_count":None, "occupied_area_m2":round(area, 2), "corridor_vehicles":corridor}

    def update_discharge_tripwire(
        self,
        discharge_vehicles: List[TrackedVehicle],
    ) -> int:
        """
        Monitors Kamera Akuisisi stop-line tripwire to count discharged vehicles (N_served).

        Args:
            discharge_vehicles: Tracked vehicles detected in Kamera Akuisisi.

        Returns:
            Cumulative count of vehicles served during this green cycle.
        """
        for v in discharge_vehicles:
            tid = v.track_id
            curr_y = v.bottom_center[1]

            if tid in self.tripwire_history:
                prev_y = self.tripwire_history[tid]
                # Vehicle crossed downward across the tripwire line
                if prev_y < self.stop_line_tripwire_y <= curr_y:
                    if tid not in self.served_track_ids:
                        self.served_track_ids.add(tid)
                        self.n_served += 1

            self.tripwire_history[tid] = curr_y

        return self.n_served

    def reset_discharge_cycle(self) -> int:
        """Resets the served vehicle counter at the onset of a new green phase."""
        last_served = self.n_served
        self.n_served = 0
        self.served_track_ids.clear()
        self.tripwire_history.clear()
        return last_served

    def draw_metrics_overlay(
        self,
        frame: np.ndarray,
        metrics: Dict[str, Any],
        t_anfis: Optional[float] = None,
    ) -> np.ndarray:
        """Renders HUD with PKJI traffic parameters and ANFIS decision on video feed."""
        # Top-left HUD card
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (480, 210), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (20, 20), (480, 210), (0, 200, 255), 2)

        # Title
        cv2.putText(
            frame,
            "ATSC EDGE PERCEPTION ENGINE",
            (35, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Metrics rows
        vw_str = f"Weighted Flow (V_w) : {metrics['V_w']:.2f} PCU (MC:{metrics['n_mc']}, LV:{metrics['n_lv']}, HV:{metrics['n_hv']})"
        q_str = f"Queue Length  (Q)   : {metrics['Q']:.1f} m ({metrics['stopped_vehicles_count']} stopped)"
        l_str = f"Lane Occupancy (L)  : {metrics['L']:.1f}% ({metrics['occupied_area_m2']:.1f} m2)"

        cv2.putText(frame, vw_str, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, q_str, (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, l_str, (35, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if t_anfis is not None:
            anfis_str = f"ANFIS Green Time : {t_anfis:.1f} s"
            cv2.putText(frame, anfis_str, (35, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            status_str = "Phase Status      : APPROACH MONITORING"
            cv2.putText(frame, status_str, (35, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame
