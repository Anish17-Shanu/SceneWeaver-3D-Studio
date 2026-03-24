import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const fileInput = document.getElementById("file-input");
const titleInput = document.getElementById("scene-title");
const buildStatus = document.getElementById("build-status");
const sceneMeta = document.getElementById("scene-meta");
const savedScenesNode = document.getElementById("saved-scenes");
const analysisGrid = document.getElementById("analysis-grid");
const buildButton = document.getElementById("build-scene");
const exportButton = document.getElementById("export-scene");
const refreshScenesButton = document.getElementById("refresh-scenes");
const toggleSpinButton = document.getElementById("toggle-spin");
const resetCameraButton = document.getElementById("reset-camera");

const sceneStage = document.getElementById("scene-canvas");
const previewCanvas = document.getElementById("preview-canvas");
const previewCtx = previewCanvas.getContext("2d");

const densityInput = document.getElementById("density");
const depthStrengthInput = document.getElementById("depth-strength");
const pointSizeInput = document.getElementById("point-size");
const videoLayersInput = document.getElementById("video-layers");
const layerOffsetInput = document.getElementById("layer-offset");
const depthModeInput = document.getElementById("depth-mode");

const state = {
  sourceFile: null,
  sourceType: null,
  renderedPointCount: 0,
  analysis: null,
  autoRotate: true,
};

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setSize(sceneStage.clientWidth, sceneStage.clientHeight);
sceneStage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07111d);

const camera = new THREE.PerspectiveCamera(58, sceneStage.clientWidth / sceneStage.clientHeight, 0.1, 5000);
camera.position.set(0, 0, 520);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const ambient = new THREE.AmbientLight(0xffffff, 0.8);
const directional = new THREE.DirectionalLight(0x7de4ff, 1.2);
directional.position.set(120, 100, 220);
scene.add(ambient, directional);

const grid = new THREE.GridHelper(800, 16, 0x1c4468, 0x0e243b);
grid.position.y = 180;
scene.add(grid);

let pointCloud = null;
let cameraTrackLine = null;

function getControlsPayload() {
  return {
    density: Number(densityInput.value),
    depthMode: depthModeInput.value,
    depthStrength: Number(depthStrengthInput.value),
    pointSize: Number(pointSizeInput.value),
    videoLayers: Number(videoLayersInput.value),
    layerOffset: Number(layerOffsetInput.value),
  };
}

function clearSceneObjects() {
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }
  if (cameraTrackLine) {
    scene.remove(cameraTrackLine);
    cameraTrackLine.geometry.dispose();
    cameraTrackLine.material.dispose();
    cameraTrackLine = null;
  }
}

function rgbaToColor(rgba) {
  const match = rgba.match(/rgba?\((\d+), (\d+), (\d+)/);
  if (!match) return new THREE.Color(0xffffff);
  return new THREE.Color(`rgb(${match[1]}, ${match[2]}, ${match[3]})`);
}

function renderPointCloud(points, pointSize) {
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);

  points.forEach((point, index) => {
    positions[index * 3] = point.x;
    positions[index * 3 + 1] = -point.y;
    positions[index * 3 + 2] = point.z;

    const color = rgbaToColor(point.color);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();

  const material = new THREE.PointsMaterial({
    size: Math.max(1.2, pointSize * 1.8),
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    sizeAttenuation: true,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
}

function renderCameraTrack(cameraTrack) {
  if (!cameraTrack || cameraTrack.length < 2) return;
  const points = cameraTrack.map((point) => new THREE.Vector3(point.x * 28, -point.y * 28, point.z * 28));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x7dffc9 });
  cameraTrackLine = new THREE.Line(geometry, material);
  scene.add(cameraTrackLine);
}

async function drawSourcePreview(file) {
  const url = URL.createObjectURL(file);

  if (file.type.startsWith("video/")) {
    const video = document.createElement("video");
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = reject;
    });
    video.currentTime = 0;
    await new Promise((resolve) => {
      video.onseeked = resolve;
    });
    previewCanvas.width = video.videoWidth || 320;
    previewCanvas.height = video.videoHeight || 180;
    previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
  } else {
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
    previewCanvas.width = image.width;
    previewCanvas.height = image.height;
    previewCtx.drawImage(image, 0, 0);
  }
}

function renderAnalysis(analysis) {
  if (!analysis) {
    analysisGrid.innerHTML = "";
    return;
  }
  const keyframeCount = analysis.keyframes ? analysis.keyframes.length : 0;
  const cameraSteps = analysis.cameraTrack ? Math.max(analysis.cameraTrack.length - 1, 0) : 0;
  analysisGrid.innerHTML = [
    ["Depth Mode", analysis.mode],
    ["Layers", analysis.layers ?? 1],
    ["Sampled Frames", analysis.sampledFrames ?? 1],
    ["Keyframes", keyframeCount],
    ["Camera Steps", cameraSteps],
  ]
    .map(([label, value]) => `<article><strong>${label}</strong><div>${value}</div></article>`)
    .join("");
}

async function buildSceneFromSource() {
  if (!state.sourceFile) {
    buildStatus.textContent = "Choose an image or video first.";
    return;
  }

  buildStatus.textContent = "Uploading source and reconstructing scene...";
  await drawSourcePreview(state.sourceFile);

  const formData = new FormData();
  formData.append("file", state.sourceFile);
  formData.append("controls", JSON.stringify(getControlsPayload()));

  const response = await fetch("/api/reconstruct", {
    method: "POST",
    body: formData,
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "Reconstruction failed");
  }

  clearSceneObjects();
  state.sourceType = result.sourceType;
  state.renderedPointCount = result.pointCount;
  state.analysis = result.analysis;

  renderPointCloud(result.points, result.controls.pointSize);
  renderCameraTrack(result.analysis.cameraTrack);
  renderAnalysis(result.analysis);

  sceneMeta.textContent = `${result.sourceType} scene | ${result.pointCount} points | ${result.analysis.mode}`;
  buildStatus.textContent = `Reconstruction complete using ${result.analysis.mode} depth.`;
}

async function exportScene() {
  if (!state.renderedPointCount) {
    buildStatus.textContent = "Build a scene before exporting.";
    return;
  }

  const payload = {
    title: titleInput.value.trim() || `${state.sourceType || "scene"} reconstruction`,
    sourceType: state.sourceType || "image",
    pointCount: state.renderedPointCount,
    controls: getControlsPayload(),
    analysis: state.analysis || {},
  };

  const response = await fetch("/api/scenes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    buildStatus.textContent = result.error || "Export failed.";
    return;
  }
  buildStatus.textContent = `Scene exported: ${result.title}`;
  await loadSavedScenes();
}

async function loadSavedScenes() {
  const response = await fetch("/api/scenes");
  const payload = await response.json();
  const items = payload.items || [];
  savedScenesNode.innerHTML = items.length
    ? items
        .map(
          (item) =>
            `<article><strong>${item.title}</strong><span>${item.sourceType} | ${item.pointCount} points</span><span>${item.analysis?.mode || "scene-export"} | ${item.createdAt}</span></article>`
        )
        .join("")
    : `<article><strong>No saved scenes yet</strong><span>Export a generated scene to persist it.</span></article>`;
}

function animate() {
  requestAnimationFrame(animate);
  controls.autoRotate = state.autoRotate;
  controls.autoRotateSpeed = 0.9;
  controls.update();
  renderer.render(scene, camera);
}

function handleResize() {
  const width = sceneStage.clientWidth;
  const height = sceneStage.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", handleResize);

fileInput.addEventListener("change", () => {
  state.sourceFile = fileInput.files?.[0] || null;
  buildStatus.textContent = state.sourceFile ? `Loaded ${state.sourceFile.name}` : "Load an image or video to begin.";
});

buildButton.addEventListener("click", () => {
  buildSceneFromSource().catch((error) => {
    buildStatus.textContent = error.message || "Unable to build scene.";
  });
});

exportButton.addEventListener("click", () => {
  exportScene().catch((error) => {
    buildStatus.textContent = error.message || "Unable to export scene.";
  });
});

refreshScenesButton.addEventListener("click", loadSavedScenes);

toggleSpinButton.addEventListener("click", () => {
  state.autoRotate = !state.autoRotate;
});

resetCameraButton.addEventListener("click", () => {
  camera.position.set(0, 0, 520);
  controls.target.set(0, 0, 0);
  controls.update();
});

loadSavedScenes();
renderAnalysis(null);
animate();
