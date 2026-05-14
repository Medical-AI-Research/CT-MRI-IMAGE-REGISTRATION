# CT/MRI Registration — Web App

## File layout

```
project/
├── pipeline.py        ← all processing logic
├── app.py             ← FastAPI server
├── index.html         ← frontend (open in browser directly)
├── requirements.txt
├── Dockerfile
└── jobs/              ← created automatically at runtime
```

---

## Quick Start (local, no Docker)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. SimpleITK 2.3.x ships with its own ITK — no separate ITK install needed.

### 2. Start the backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Open the frontend

Just double-click `index.html` or open it in your browser:
```
file:///C:/Users/YourName/project/index.html
```

The `API` variable at the top of the `<script>` block is already set to `http://localhost:8000`.

### 4. Use the app

1. Zip your CT DICOM folder → `ct_series.zip`
2. Zip your MRI DICOM folder → `mri_series.zip`
3. Drop them into the upload boxes
4. Click **Run Registration**
5. Wait 1–5 minutes (registration is CPU-bound)
6. View previews and download the results ZIP

---

## How to ZIP your DICOM files

**Windows (right-click):**
```
Select all DICOM files → Right-click → Send to → Compressed (zipped) folder
```

**Command line:**
```bash
# Linux / Mac
zip -r ct_series.zip /path/to/ct_dicoms/
zip -r mri_series.zip /path/to/mri_dicoms/

# Windows PowerShell
Compress-Archive -Path D:\CT\* -DestinationPath ct_series.zip
Compress-Archive -Path D:\MRI\* -DestinationPath mri_series.zip
```

The ZIP can contain the DICOM files at the top level **or** inside one subfolder — the server will find them automatically.

---

## Docker deployment

### Build and run locally

```bash
docker build -t medreg .
docker run -p 8000:8000 -v $(pwd)/jobs:/app/jobs medreg
```

### Deploy to a VPS (DigitalOcean / Linode / Hetzner)

```bash
# On your VPS:
git clone <your-repo>
cd project
docker build -t medreg .
docker run -d \
  --name medreg \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /data/medreg-jobs:/app/jobs \
  medreg
```

Then open port 8000 in your firewall and update `index.html`:

```js
// Change this line in index.html:
const API = 'http://YOUR_SERVER_IP:8000';
```

### Recommended VPS specs

| Use case | RAM | CPU | Storage |
|---|---|---|---|
| Dev / testing | 4 GB | 2 vCPU | 50 GB |
| Production | 8–16 GB | 4 vCPU | 100 GB |

---

## What's in the results ZIP?

```
results/
├── registered/
│   ├── ct_fixed.nii.gz             ← original CT (reoriented)
│   ├── mri_original.nii.gz         ← original MRI (reoriented)
│   ├── mri_rigid_to_ct.nii.gz      ← MRI registered to CT grid
│   ├── ct_cropped.nii.gz           ← CT cropped to MRI FOV
│   ├── mri_cropped.nii.gz          ← MRI cropped to MRI FOV
│   ├── ct_mask_auto.nii.gz         ← auto bone mask (HU > 150)
│   ├── rigid_transform.tfm         ← ITK transform (use in 3D Slicer)
│   ├── fov_bounds.json             ← crop bounding box
│   └── metrics.json                ← all computed metrics
└── previews/
    ├── ct_mid.png
    ├── mri_mid.png
    ├── overlay_mid.png
    ├── checkerboard_mid.png
    ├── edge_overlay_mid.png
    └── organ_overlap_mid.png
```

---

## Understanding the metrics

| Metric | What it measures | Good value |
|---|---|---|
| NMI | Normalized Mutual Information — intensity correlation | > 1.3 |
| Edge Dice | Overlap of CT and MRI edges | > 0.4 |
| FOV Overlap | How much of CT is covered by MRI | context-dependent |
| NCC | Normalized Cross-Correlation | > 0.6 |
| MAD | Mean Absolute Difference (intensity) | < 0.2 |
| Mismatch (1-NCC) | Complement of NCC | < 0.4 |

---

## Troubleshooting

**"No DICOM series found"**
→ Make sure your ZIP contains `.dcm` files (or files without extension that are DICOM). Do not zip a folder-of-folders more than one level deep.

**Registration takes too long / times out**
→ Normal on a slow machine. The `--reload` flag in uvicorn does not affect timeout. Registration takes 1–5 min for typical head/abdomen volumes.

**CORS error in browser console**
→ The backend already has `allow_origins=["*"]`. If you still see CORS errors, check that the `API` variable in `index.html` matches the exact address/port of your server.

**Large files fail to upload**
→ Add `--limit-request-body 2147483648` (2 GB) to the uvicorn command.

**Out of memory**
→ SimpleITK loads the full volume. A typical CT is 200–500 MB in memory. Use a machine with at least 4 GB RAM free.