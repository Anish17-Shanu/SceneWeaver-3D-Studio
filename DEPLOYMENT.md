# Deployment

## Local

```bash
python -m pip install -r requirements.txt
python server/preload_midas.py
python server/app.py
```

Open:

```text
http://127.0.0.1:8095
```

## Docker

```bash
docker build -t sceneweaver-3d-studio .
docker run -p 8095:8095 sceneweaver-3d-studio
```

## Production notes

- move saved scene metadata into a database
- add authenticated uploads and project ownership
- add camera pose estimation and temporal fusion beyond the current MiDaS-assisted depth layer
- move the renderer to WebGL for larger scenes
