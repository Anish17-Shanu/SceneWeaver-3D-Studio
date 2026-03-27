"""SceneWeaver 3D Studio server created by Anish Kumar."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from model_depth import midas_available
from reconstruction import parse_controls, reconstruct_file


ROOT = Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "client"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = ROOT / "uploads"
SCENES_FILE = DATA_DIR / "generated-scenes.json"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


def load_scenes():
    with SCENES_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_scenes(payload):
    with SCENES_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def classify_source(filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Unsupported file type. Use an image or video file.")


app = Flask(__name__, static_folder=str(CLIENT_DIR), static_url_path="")


@app.get("/api/health")
def health():
    scenes = load_scenes()
    return jsonify(
        {
            "service": "SceneWeaver 3D Studio",
            "status": "ok",
            "savedScenes": len(scenes.get("items", [])),
            "reconstructionMode": "cv-backed",
            "midasAvailable": midas_available(),
        }
    )


@app.get("/api/scenes")
def list_scenes():
    return jsonify(load_scenes())


@app.post("/api/scenes")
def create_scene():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    source_type = str(payload.get("sourceType", "")).strip().lower()
    points = int(payload.get("pointCount", 0))
    controls = payload.get("controls", {})
    analysis = payload.get("analysis", {})

    if not title:
        return jsonify({"error": "title is required"}), 400
    if source_type not in {"image", "video"}:
        return jsonify({"error": "sourceType must be image or video"}), 400
    if points <= 0:
        return jsonify({"error": "pointCount must be positive"}), 400

    scenes = load_scenes()
    record = {
        "id": str(uuid4()),
        "title": title,
        "sourceType": source_type,
        "pointCount": points,
        "controls": controls,
        "analysis": analysis,
        "createdAt": datetime.utcnow().isoformat() + "Z",
    }
    scenes["items"].insert(0, record)
    save_scenes(scenes)
    return jsonify(record), 201


@app.post("/api/reconstruct")
def reconstruct():
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "filename is required"}), 400

    try:
        source_type = classify_source(uploaded_file.filename)
        controls = parse_controls(request.form.get("controls"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    filename = secure_filename(uploaded_file.filename)
    stored_name = f"{uuid4()}-{filename}"
    stored_path = UPLOADS_DIR / stored_name
    uploaded_file.save(stored_path)

    try:
        result = reconstruct_file(stored_path, source_type, controls)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "sourceType": result["sourceType"],
            "points": result["points"],
            "pointCount": len(result["points"]),
            "previewWidth": result["previewWidth"],
            "previewHeight": result["previewHeight"],
            "controls": controls,
            "analysis": result["analysis"],
        }
    )


@app.get("/")
def index():
    return send_from_directory(CLIENT_DIR, "index.html")


@app.get("/<path:path>")
def static_proxy(path: str):
    return send_from_directory(CLIENT_DIR, path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8095, debug=True)