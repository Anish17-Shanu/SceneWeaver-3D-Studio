from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from model_depth import estimate_depth_map, midas_available


def _clamp_int(value, default, minimum, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def parse_controls(raw_controls):
    if isinstance(raw_controls, str):
        try:
            raw_controls = json.loads(raw_controls)
        except json.JSONDecodeError:
            raw_controls = {}
    raw_controls = raw_controls or {}
    depth_mode = str(raw_controls.get("depthMode", "auto")).strip().lower() or "auto"
    if depth_mode not in {"auto", "heuristic", "midas"}:
        depth_mode = "auto"
    return {
        "density": _clamp_int(raw_controls.get("density"), 8, 3, 20),
        "depthStrength": _clamp_int(raw_controls.get("depthStrength"), 90, 20, 220),
        "pointSize": _clamp_int(raw_controls.get("pointSize"), 2, 1, 8),
        "videoLayers": _clamp_int(raw_controls.get("videoLayers"), 6, 2, 12),
        "layerOffset": _clamp_int(raw_controls.get("layerOffset"), 30, 6, 120),
        "depthMode": depth_mode,
    }


def _resize_frame(frame, width=320):
    height = max(120, int(frame.shape[0] * (width / frame.shape[1])))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _resolve_depth_mode(preferred_mode):
    if preferred_mode == "heuristic":
        return "heuristic"
    if preferred_mode == "midas":
        return "midas"
    return "midas" if midas_available() else "heuristic"


def _heuristic_depth(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_32F)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    grad_norm = cv2.normalize(gradient, None, 0, 1.0, cv2.NORM_MINMAX)
    lap_norm = cv2.normalize(np.abs(laplacian), None, 0, 1.0, cv2.NORM_MINMAX)
    brightness = gray.astype(np.float32) / 255.0
    return (1.0 - brightness) * 0.7 + grad_norm * 0.23 + lap_norm * 0.07


def _frame_to_points(frame_bgr, controls, layer_index=0, total_layers=1, motion_map=None):
    frame = _resize_frame(frame_bgr)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_mode = _resolve_depth_mode(controls["depthMode"])
    motion = motion_map if motion_map is not None else np.zeros(frame.shape[:2], dtype=np.float32)

    if depth_mode == "midas":
        try:
            base_depth = estimate_depth_map(frame)
        except Exception:
            depth_mode = "heuristic"
            base_depth = _heuristic_depth(frame)
    else:
        base_depth = _heuristic_depth(frame)

    base_depth = cv2.normalize(base_depth.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
    height, width = base_depth.shape
    step = controls["density"]
    point_size = controls["pointSize"]
    layer_shift = (layer_index - (total_layers - 1) / 2.0) * controls["layerOffset"]

    points = []
    for y in range(2, height - 2, step):
        for x in range(2, width - 2, step):
            depth = base_depth[y, x] * controls["depthStrength"] + motion[y, x] * 42.0 + layer_shift
            r, g, b = rgb[y, x]
            points.append(
                {
                    "x": round((x - width / 2.0) * 2.3, 3),
                    "y": round((y - height / 2.0) * 2.3, 3),
                    "z": round(depth - controls["depthStrength"] / 2.0, 3),
                    "size": point_size,
                    "color": f"rgba({int(r)}, {int(g)}, {int(b)}, 0.88)",
                }
            )

    preview = Image.fromarray(rgb)
    return {
        "points": points,
        "previewWidth": preview.width,
        "previewHeight": preview.height,
        "depthModeResolved": depth_mode,
    }


def reconstruct_image(file_path: Path, controls):
    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError("Unable to read image file")
    result = _frame_to_points(image, controls, 0, 1)
    return {
        "sourceType": "image",
        "points": result["points"],
        "previewWidth": result["previewWidth"],
        "previewHeight": result["previewHeight"],
        "analysis": {
            "mode": result["depthModeResolved"],
            "layers": 1,
            "sampledFrames": 1,
        },
    }


def reconstruct_video(file_path: Path, controls):
    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        raise ValueError("Unable to open video file")

    frame_total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), controls["videoLayers"])
    sample_count = controls["videoLayers"]
    sample_indices = sorted({min(frame_total - 1, int((frame_total - 1) * i / max(sample_count - 1, 1))) for i in range(sample_count)})

    points = []
    preview_width = 320
    preview_height = 180
    previous_gray = None
    previous_keypoints = None
    previous_descriptors = None
    resolved_mode = _resolve_depth_mode(controls["depthMode"])
    orb = cv2.ORB_create(1200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    camera_track = [{"x": 0.0, "y": 0.0, "z": 0.0}]
    keyframes = []

    for layer_index, frame_index in enumerate(sample_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        resized = _resize_frame(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        motion_map = np.zeros_like(gray, dtype=np.float32)
        frame_motion_score = 0.0
        if previous_gray is not None and previous_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, previous_gray)
            motion_map = cv2.normalize(diff.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
            frame_motion_score = float(np.mean(diff))
        previous_gray = gray

        keypoints, descriptors = orb.detectAndCompute(gray, None)
        estimated_translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if (
            previous_descriptors is not None
            and descriptors is not None
            and len(descriptors) > 8
            and len(previous_descriptors) > 8
            and previous_keypoints is not None
        ):
            matches = matcher.match(previous_descriptors, descriptors)
            matches = sorted(matches, key=lambda item: item.distance)[:160]
            if len(matches) >= 8:
                pts_prev = np.float32([previous_keypoints[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])
                height, width = gray.shape
                focal = max(width, height)
                principal = (width / 2.0, height / 2.0)
                camera_matrix = np.array(
                    [
                        [focal, 0, principal[0]],
                        [0, focal, principal[1]],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                essential, _ = cv2.findEssentialMat(
                    pts_curr,
                    pts_prev,
                    camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0,
                )
                if essential is not None:
                    _, _, translation, _ = cv2.recoverPose(essential, pts_curr, pts_prev, camera_matrix)
                    estimated_translation = translation.flatten().astype(np.float32)
                    estimated_translation /= max(np.linalg.norm(estimated_translation), 1.0)

        camera_position = camera_position + estimated_translation
        camera_track.append(
            {
                "x": round(float(camera_position[0]) * 30.0, 4),
                "y": round(float(camera_position[1]) * 30.0, 4),
                "z": round(float(camera_position[2]) * 30.0, 4),
            }
        )

        if layer_index == 0 or frame_motion_score > 18.0 or np.linalg.norm(estimated_translation) > 0.2:
            keyframes.append(
                {
                    "frameIndex": int(frame_index),
                    "motionScore": round(frame_motion_score, 3),
                    "cameraStep": {
                        "x": round(float(estimated_translation[0]), 4),
                        "y": round(float(estimated_translation[1]), 4),
                        "z": round(float(estimated_translation[2]), 4),
                    },
                }
            )

        previous_keypoints = keypoints
        previous_descriptors = descriptors

        result = _frame_to_points(frame, controls, layer_index, len(sample_indices), motion_map=motion_map)
        preview_width = result["previewWidth"]
        preview_height = result["previewHeight"]
        resolved_mode = result["depthModeResolved"]
        points.extend(result["points"])

    capture.release()

    if not points:
        raise ValueError("No usable frames were extracted from the video")

    return {
        "sourceType": "video",
        "points": points,
        "previewWidth": preview_width,
        "previewHeight": preview_height,
        "analysis": {
            "mode": resolved_mode,
            "layers": len(sample_indices),
            "sampledFrames": len(sample_indices),
            "cameraTrack": camera_track,
            "keyframes": keyframes,
        },
    }


def reconstruct_file(file_path: Path, source_type: str, controls):
    if source_type == "image":
        return reconstruct_image(file_path, controls)
    if source_type == "video":
        return reconstruct_video(file_path, controls)
    raise ValueError("Unsupported source type")
