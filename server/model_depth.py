from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_NAME = "MiDaS_small"
MODEL_REPO = "intel-isl/MiDaS"
TORCH_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / ".torch-cache"
LOCAL_MIDAS_REPO = TORCH_CACHE_DIR / "intel-isl_MiDaS_master"


@lru_cache(maxsize=1)
def get_midas_bundle():
    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_CACHE_DIR))
    if LOCAL_MIDAS_REPO.exists():
        model = torch.hub.load(str(LOCAL_MIDAS_REPO), MODEL_NAME, source="local")
        transforms = torch.hub.load(str(LOCAL_MIDAS_REPO), "transforms", source="local")
    else:
        model = torch.hub.load(MODEL_REPO, MODEL_NAME, trust_repo=True)
        transforms = torch.hub.load(MODEL_REPO, "transforms", trust_repo=True)
    transform = transforms.small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return {"model": model, "transform": transform, "device": device}


def midas_available():
    try:
        get_midas_bundle()
        return True
    except Exception:
        return False


def estimate_depth_map(frame_bgr: np.ndarray) -> np.ndarray:
    bundle = get_midas_bundle()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = bundle["transform"](frame_rgb).to(bundle["device"])

    with torch.no_grad():
        prediction = bundle["model"](input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return cv2.normalize(depth, None, 0, 1.0, cv2.NORM_MINMAX)
