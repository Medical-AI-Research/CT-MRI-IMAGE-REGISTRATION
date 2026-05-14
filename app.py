"""
app.py — FastAPI backend for CT/MRI Registration Web App
Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import zipfile
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pipeline import run_full_pipeline

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="CT/MRI Registration API")

# Allow the React/HTML frontend to call this API from any origin during dev.
# In production, replace "*" with your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built frontend (index.html + assets) from ./frontend/dist
# Comment this out if you're running frontend separately (e.g. with Vite dev server)
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

JOBS_ROOT = os.path.join(os.path.dirname(__file__), "jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

# In-memory job store. Fine for single-process dev; swap for Redis in production.
JOBS: dict[str, dict] = {}

# One background thread per job (registration is CPU-bound, not async-friendly)
executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _extract_zip(upload: bytes, dest_dir: str):
    """Write uploaded bytes to a temp zip and extract."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = dest_dir + ".zip"
    with open(zip_path, "wb") as f:
        f.write(upload)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    os.remove(zip_path)


def _find_dicom_dir(root: str) -> str:
    """
    Walk extracted folder and return the first directory that contains
    at least one .dcm file (or any file if no .dcm found at top level).
    Handles nested zips like: upload.zip → PatientFolder/ → *.dcm
    """
    for dirpath, _, filenames in os.walk(root):
        dcm_files = [f for f in filenames if not f.startswith(".")]
        if dcm_files:
            return dirpath
    return root


def _zip_results(job_dir: str, out_zip: str):
    """Zip the registered/ and previews/ sub-folders."""
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["registered", "previews"]:
            folder_path = os.path.join(job_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    arc_name = os.path.relpath(abs_path, job_dir)
                    z.write(abs_path, arc_name)


def _background_register(job_id: str, ct_dir: str, mri_dir: str, job_dir: str):
    """This runs in a thread pool."""
    try:
        JOBS[job_id]["status"] = "running"

        result = run_full_pipeline(ct_dir, mri_dir, job_dir)

        # Zip everything up for single-click download
        out_zip = os.path.join(job_dir, "results.zip")
        _zip_results(job_dir, out_zip)

        JOBS[job_id].update({
            "status":   "done",
            "metrics":  result["metrics"],
            "zip_path": out_zip,
            # relative paths for preview image endpoints
            "previews": [
                "ct_mid.png",
                "mri_mid.png",
                "overlay_mid.png",
                "checkerboard_mid.png",
                "edge_overlay_mid.png",
                "organ_overlap_mid.png",
            ],
        })

    except Exception:
        err = traceback.format_exc()
        print(f"[Job {job_id}] FAILED:\n{err}")
        JOBS[job_id] = {"status": "failed", "error": err}


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "CT/MRI Registration API is running. POST /register to start."}


@app.post("/register")
async def register(
    ct_zip:  UploadFile = File(..., description="ZIP file containing CT DICOM series"),
    mri_zip: UploadFile = File(..., description="ZIP file containing MRI DICOM series"),
):
    """
    Accept two ZIP uploads (CT DICOMs, MRI DICOMs).
    Starts registration in background. Returns job_id immediately.
    """
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

    JOBS[job_id] = {
        "status":  "queued",
        "job_id":  job_id,
        "ct_file":  ct_zip.filename,
        "mri_file": mri_zip.filename,
    }

    executor.submit(_background_register, job_id, ct_dicom_dir, mri_dicom_dir, job_dir)

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll this endpoint every few seconds after POST /register."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Don't expose internal file paths
    safe = {k: v for k, v in job.items() if k != "zip_path"}
    return JSONResponse(content=safe)


@app.get("/preview/{job_id}/{filename}")
def get_preview(job_id: str, filename: str):
    """Serve a preview PNG by job_id and filename."""
    # Whitelist filenames to prevent path traversal
    allowed = {
        "ct_mid.png", "mri_mid.png", "overlay_mid.png",
        "checkerboard_mid.png", "edge_overlay_mid.png", "organ_overlap_mid.png",
    }
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Invalid filename")

    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Preview not ready")

    path = os.path.join(JOBS_ROOT, job_id, "previews", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Preview file not found")

    return FileResponse(path, media_type="image/png")


@app.get("/download/{job_id}")
def download_results(job_id: str):
    """Download the full results ZIP (NIfTI volumes + transform + metrics + previews)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not done yet (status: {job['status']})")

    zip_path = job.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="Result ZIP not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"registration_results_{job_id[:8]}.zip",
    )


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Clean up job files and remove from memory."""
    job_dir = os.path.join(JOBS_ROOT, job_id)
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
    JOBS.pop(job_id, None)
    return {"deleted": job_id}