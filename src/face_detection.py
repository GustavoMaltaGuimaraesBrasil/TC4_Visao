"""
Helpers for face detection with MediaPipe and Haar Cascade.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import cv2
import numpy as np


def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None
    return mp


def _detect_faces_mediapipe(
    image_bgr: np.ndarray,
    mp,
    min_conf: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    with mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=min_conf,
    ) as face_detection:
        results = face_detection.process(image_rgb)

    detections = getattr(results, "detections", None) or []
    for detection in detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    return boxes


def _load_haar_classifiers() -> Tuple[Optional[cv2.CascadeClassifier], Optional[cv2.CascadeClassifier]]:
    cascade_root = cv2.data.haarcascades
    frontal_path = f"{cascade_root}haarcascade_frontalface_default.xml"
    profile_path = f"{cascade_root}haarcascade_profileface.xml"
    frontal = cv2.CascadeClassifier(frontal_path)
    profile = cv2.CascadeClassifier(profile_path)
    if frontal.empty():
        frontal = None
    if profile.empty():
        profile = None
    return frontal, profile


def _detect_faces_haar(
    image_bgr: np.ndarray,
    frontal: Optional[cv2.CascadeClassifier],
    profile: Optional[cv2.CascadeClassifier],
    scale_factor: float = 1.05,
    min_neighbors: int = 3,
    min_rel_size: float = 0.02,
    try_rotations: bool = True,
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    min_size = (max(1, int(width * min_rel_size)), max(1, int(height * min_rel_size)))
    boxes: List[Tuple[int, int, int, int]] = []

    def _run_detector(detector, img):
        if detector is None:
            return []
        return detector.detectMultiScale(
            img,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

    boxes.extend(_run_detector(frontal, gray))
    boxes.extend(_run_detector(profile, gray))

    if try_rotations:
        for angle in (90, 270):
            rotated = _rotate_image(gray, angle)
            for (x, y, w, h) in _run_detector(frontal, rotated):
                rx, ry = _rotate_coords(x, y, w, h, gray.shape, angle)
                boxes.append((rx, ry, w, h))

    return boxes


def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _rotate_coords(
    x: int,
    y: int,
    w: int,
    h: int,
    original_shape: Tuple[int, int],
    angle: int,
) -> Tuple[int, int]:
    height, width = original_shape[:2]
    if angle == 90:
        return height - y - h, x
    if angle == 270:
        return y, width - x - w
    return x, y
