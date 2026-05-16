"""
app.py — FastAPI backend for CT/MRI Registration
Serves index.html at / and API at /register, /status, /preview, /download
"""

import os
import uuid
import json
import zipfile
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_full_pipeline

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

app = FastAPI(title="CT/MRI Registration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jobs root — set via env var in Docker, falls back to local ./jobs
JOBS_ROOT = os.environ.get("JOBS_ROOT", os.path.join(os.path.dirname(__file__), "jobs"))
os.makedirs(JOBS_ROOT, exist_ok=True)

# In-memory cache — disk is source of truth
_MEM: dict[str, dict] = {}

executor = ThreadPoolExecutor(max_workers=2)

# Path to index.html (same directory as app.py)
INDEX_HTML = os.path.join(os.path.dirname(__file__), "index.html")


# ─────────────────────────────────────────────
#  DISK PERSISTENCE
# ─────────────────────────────────────────────

def _status_path(job_id: str) -> str:
    return os.path.join(JOBS_ROOT, job_id, "job_status.json")


def _save_job(job_id: str, data: dict):
    safe = {k: v for k, v in data.items() if k != "zip_path"}
    path = _status_path(job_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)
    _MEM[job_id] = data


def _load_job(job_id: str) -> dict | None:
    path = _status_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    zip_path = os.path.join(JOBS_ROOT, job_id, "results.zip")
    if os.path.exists(zip_path):
        data["zip_path"] = zip_path
    _MEM[job_id] = data
    return data


def _get_job(job_id: str) -> dict | None:
    return _MEM.get(job_id) or _load_job(job_id)


# ─────────────────────────────────────────────
#  UPLOAD HELPERS
# ─────────────────────────────────────────────

def _extract_zip(raw_bytes: bytes, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = dest_dir + ".zip"
    with open(zip_path, "wb") as f:
        f.write(raw_bytes)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    os.remove(zip_path)


def _find_dicom_dir(root: str) -> str:
    for dirpath, _, filenames in os.walk(root):
        real = [f for f in filenames if not f.startswith(".") and not f.startswith("__")]
        if real:
            return dirpath
    return root


def _zip_results(job_dir: str, out_zip: str):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["registered", "previews"]:
            fp = os.path.join(job_dir, folder)
            if not os.path.isdir(fp):
                continue
            for root, _, files in os.walk(fp):
                for fname in files:
                    abs_p = os.path.join(root, fname)
                    z.write(abs_p, os.path.relpath(abs_p, job_dir))


# ─────────────────────────────────────────────
#  BACKGROUND RUNNER
# ─────────────────────────────────────────────

def _run(job_id: str, ct_dir: str, mri_dir: str, job_dir: str):
    try:
        _save_job(job_id, {**_get_job(job_id), "status": "running"})
        result = run_full_pipeline(ct_dir, mri_dir, job_dir)

        out_zip = os.path.join(job_dir, "results.zip")
        _zip_results(job_dir, out_zip)

        _save_job(job_id, {
            "status":   "done",
            "job_id":   job_id,
            "metrics":  result["metrics"],
            "zip_path": out_zip,
            "previews": [
                "ct_mid.png", "mri_mid.png", "overlay_mid.png",
                "checkerboard_mid.png", "edge_overlay_mid.png", "organ_overlap_mid.png",
            ],
        })

    except Exception:
        err = traceback.format_exc()
        print(f"[Job {job_id}] FAILED:\n{err}")
        _save_job(job_id, {"status": "failed", "job_id": job_id, "error": err})


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend index.html directly from the API."""
    if os.path.exists(INDEX_HTML):
        with open(INDEX_HTML) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>CT/MRI Registration API running</h1><p>index.html not found.</p>")


@app.post("/register")
async def register(
    ct_zip:  UploadFile = File(...),
    mri_zip: UploadFile = File(...),
):
    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    ct_upload_dir  = os.path.join(job_dir, "uploads", "ct")
    mri_upload_dir = os.path.join(job_dir, "uploads", "mri")

    try:
        ct_bytes  = await ct_zip.read()
        mri_bytes = await mri_zip.read()
        _extract_zip(ct_bytes,  ct_upload_dir)
        _extract_zip(mri_bytes, mri_upload_dir)
        ct_dicom_dir  = _find_dicom_dir(ct_upload_dir)
        mri_dicom_dir = _find_dicom_dir(mri_upload_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploads: {e}")

    _save_job(job_id, {"status": "queued", "job_id": job_id})
    executor.submit(_run, job_id, ct_dicom_dir, mri_dicom_dir, job_dir)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({k: v for k, v in job.items() if k != "zip_path"})


@app.get("/preview/{job_id}/{filename}")
def get_preview(job_id: str, filename: str):
    allowed = {
        "ct_mid.png", "mri_mid.png", "overlay_mid.png",
        "checkerboard_mid.png", "edge_overlay_mid.png", "organ_overlap_mid.png",
    }
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Invalid filename")
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Preview not ready")
    path = os.path.join(JOBS_ROOT, job_id, "previews", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/png")


@app.get("/download/{job_id}")
def download(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Not ready: {job['status']}")
    zip_path = job.get("zip_path") or os.path.join(JOBS_ROOT, job_id, "results.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="Result ZIP missing")
    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"registration_{job_id[:8]}.zip")


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    job_dir = os.path.join(JOBS_ROOT, job_id)
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
    _MEM.pop(job_id, None)
    return {"deleted": job_id}