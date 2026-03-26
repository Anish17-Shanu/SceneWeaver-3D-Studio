# SceneWeaver 3D Studio

## Creator

This project was created, written, and maintained by **Anish Kumar (ANISH KUMAR)**.
All primary documentation in this README is presented as the work of **Anish Kumar**.

SceneWeaver 3D Studio is an image-and-video-to-3D simulation project. It accepts a single image or a video clip, runs server-side computer-vision analysis, and generates an interactive pseudo-3D point-cloud scene you can orbit, zoom, animate, and export.

## What this project does well

- Accepts both image and video inputs
- Builds a navigable 3D scene from server-side CV cues and multi-frame analysis
- Supports scene tuning, camera orbit, camera-track inspection, and export
- Provides a clean architecture to upgrade later with real monocular depth, SfM, Gaussian splatting, or NeRF pipelines

## Honest technical note

This version is a stronger CV-backed MVP with optional MiDaS depth estimation, but not yet a photoreal reconstruction stack. True industry-grade 3D scene recovery from arbitrary image or video inputs usually needs dedicated CV/ML models, camera-pose estimation, and heavier pipelines. This repo now focuses on:

- interactive reconstruction UX
- server-side pseudo-depth estimation
- optional MiDaS depth inference
- edge-aware point cloud generation
- layered video frame reconstruction with motion-aware depth bias
- keyframe selection and approximate camera-track recovery for videos
- exportable scene manifests
- future-ready architecture

## Stack

- Backend: Flask
- Frontend: HTML, CSS, vanilla JavaScript, Three.js WebGL viewer
- CV: OpenCV, NumPy, Pillow
- ML Runtime: PyTorch, TorchVision, timm
- Storage: JSON manifests on disk

## Run locally

```bash
cd "d:\Project\SceneWeaver-3D-Studio"
python -m pip install -r requirements.txt
python server/app.py
```

Open:

```text
http://127.0.0.1:8095
```

Optional model preload:

```bash
python server/preload_midas.py
```

## Features

- Upload image or video
- Generate CV-backed 3D point-cloud reconstruction
- Tune density, depth strength, point size, and video frame layers
- Choose `Auto`, `MiDaS`, or `Heuristic` depth mode
- Orbit, pan, zoom, auto-rotate, and inspect camera-track motion
- Export generated scene metadata to the backend
- Reload saved scene manifests from disk
- Process uploads on the backend through `/api/reconstruct`

## API

- `GET /api/health`
- `GET /api/scenes`
- `POST /api/reconstruct`
- `POST /api/scenes`

## Future upgrades

- Improve camera track recovery into fuller pose estimation and temporal fusion
- Add mesh generation and textured surfaces
- Add true measurement and object segmentation workflows
- Add surface reconstruction on top of the current point cloud viewer
