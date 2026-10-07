"""
Sidewalk-Curb Perspective Projection & Inverse Perspective Mapping (IPM) Engine.

Projects vehicle bumper contact points from the 12-15m oblique flank camera vantage
onto the physical road ground metric plane (10.5m corridor width x 45.0m approach length).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class HomographyEngine:
    """
    Computes and manages perspective projection matrices between camera pixel
    coordinates and physical metric ground plane coordinates.
    """

    def __init__(
        self,
        src_points_pixel: List[List[float]],
        dst_points_ground: List[List[float]],
        ground_width_m: float = 10.5,
        ground_length_m: float = 45.0,
        corridor_polygon_pixel: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Args:
            src_points_pixel: 4 curb control points in rectified 1080p pixel space [[u, v], ...].
            dst_points_ground: 4 corresponding metric coordinates on ground plane [[X, Y], ...].
            ground_width_m: Physical road corridor width in meters (default 10.5m).
            ground_length_m: Physical road approach corridor length in meters (default 45.0m).
            corridor_polygon_pixel: Optional arbitrary N-point polygon [[u, v], ...] (N >= 3)
                for road approach corridor containment test. Defaults to src_points_pixel.
        """
        self.src_points = np.array(src_points_pixel, dtype=np.float32)
        self.dst_points = np.array(dst_points_ground, dtype=np.float32)
        self.ground_width = float(ground_width_m)
        self.ground_length = float(ground_length_m)

        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            raise ValueError("Homography requires exactly 4 control point correspondences.")

        # Compute Forward and Inverse Homography Matrices
        self.H: np.ndarray = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv: np.ndarray = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # Polygon of approach corridor in pixel space for containment test (supports N >= 3 points)
        if corridor_polygon_pixel is not None and len(corridor_polygon_pixel) >= 3:
            self.corridor_poly: np.ndarray = np.array(corridor_polygon_pixel, dtype=np.int32)
        else:
            self.corridor_poly: np.ndarray = self.src_points.astype(np.int32)

    @classmethod
    def from_config(cls, config_path_or_dict: str | Dict[str, Any]) -> HomographyEngine:
        """Loads homography configuration from JSON file or dictionary."""
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = config_path_or_dict

        homo_cfg = cfg.get("homography", {})
        src = homo_cfg.get("src_points_pixel")
        dst = homo_cfg.get("dst_points_ground")
        dims = homo_cfg.get("ground_dimensions_m", {"width": 10.5, "length": 45.0})
        poly = homo_cfg.get("corridor_polygon_pixel") or cfg.get("lane_metrics", {}).get("corridor_polygon")

        if not src or not dst:
            raise ValueError("Invalid homography section in configuration.")

        return cls(
            src_points_pixel=src,
            dst_points_ground=dst,
            ground_width_m=float(dims.get("width", 10.5)),
            ground_length_m=float(dims.get("length", 45.0)),
            corridor_polygon_pixel=poly,
        )

    def measure_pixel_distance(
        self,
        p1: Tuple[float, float] | List[float],
        p2: Tuple[float, float] | List[float],
    ) -> float:
        """
        Calculates ground distance in meters between two image pixel points.
        """
        g1 = self.pixel_to_ground(float(p1[0]), float(p1[1]))
        g2 = self.pixel_to_ground(float(p2[0]), float(p2[1]))
        return float(np.hypot(g2[0] - g1[0], g2[1] - g1[1]))

    def calibrate_from_sample_ruler(
        self,
        p1_pixel: Tuple[float, float] | List[float],
        p2_pixel: Tuple[float, float] | List[float],
        real_distance_m: float,
    ) -> float:
        """
        Scales the ground metric space so that the measured distance between
        p1 and p2 in pixel space matches real_distance_m.

        Returns:
            Scale factor applied to ground dimensions and homography matrix.
        """
        if real_distance_m <= 0:
            raise ValueError("Sample ruler length must be positive.")

        current_dist = self.measure_pixel_distance(p1_pixel, p2_pixel)
        if current_dist < 1e-4:
            raise ValueError("Sample ruler points are too close together to calibrate scale.")

        scale_factor = float(real_distance_m / current_dist)

        # Scale ground dimensions and destination control points
        self.ground_width *= scale_factor
        self.ground_length *= scale_factor
        self.dst_points = self.dst_points * scale_factor

        # Scale forward and inverse Homography: H_new = S * H, H_inv_new = H_inv * S^-1
        s_mat = np.array([
            [scale_factor, 0.0, 0.0],
            [0.0, scale_factor, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        s_inv_mat = np.array([
            [1.0 / scale_factor, 0.0, 0.0],
            [0.0, 1.0 / scale_factor, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.H = np.dot(s_mat, self.H)
        self.H_inv = np.dot(self.H_inv, s_inv_mat)

        return scale_factor

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        Projects an image pixel coordinate (u, v) onto the physical ground plane.

        Args:
            u: Horizontal pixel coordinate (x).
            v: Vertical pixel coordinate (y).

        Returns:
            Tuple (X_m, Y_m) where:
                X_m is lateral position in meters [0.0, 10.5].
                Y_m is distance from stop line in meters [0.0, 45.0].
        """
        pt = np.array([[[u, v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        gx = float(transformed[0, 0, 0])
        gy = float(transformed[0, 0, 1])
        return round(gx, 3), round(gy, 3)

    def ground_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Projects a physical ground coordinate (X_m, Y_m) back onto the image plane.
        """
        pt = np.array([[[x_m, y_m]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_inv)
        u = int(round(transformed[0, 0, 0]))
        v = int(round(transformed[0, 0, 1]))
        return u, v

    def is_in_approach_corridor(self, u: float, v: float) -> bool:
        """
        Tests if a pixel point falls within the approach corridor polygon
        using cv2.pointPolygonTest.
        """
        dist = cv2.pointPolygonTest(self.corridor_poly, (float(u), float(v)), False)
        return dist >= 0

    def draw_corridor_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Visualizes the approach corridor polygon and ground distance markers."""
        # Draw corridor boundary polygon
        cv2.polylines(frame, [self.corridor_poly], isClosed=True, color=(0, 255, 255), thickness=2)

        # Draw distance tick marks along corridor (every 10 meters)
        for y_m in [10.0, 20.0, 30.0, 40.0]:
            p_left = self.ground_to_pixel(0.0, y_m)
            p_right = self.ground_to_pixel(self.ground_width, y_m)
            cv2.line(frame, p_left, p_right, (0, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{int(y_m)}m",
                (p_left[0] + 5, p_left[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 200, 200),
                1,
            )

        return frame
