---
title: CT MRI Registration
emoji: 🩻
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# CT/MRI Image Registration

A full-stack web application for multimodal medical image registration.

Upload CT and MRI DICOM series (as ZIP files), perform 3-D rigid registration, and download aligned NIfTI volumes.

## Features
- 3-D Rigid Registration (SimpleITK, Mattes Mutual Information)
- Automatic FOV Crop to MRI field-of-view
- 6 Alignment Metrics (NMI, Edge Dice, NCC, MAD, FOV Ratio, Mismatch)
- 6 Preview Images (overlay, checkerboard, edge overlay, bone mask)
- One-click ZIP download (NIfTI volumes + ITK transform + metrics)

## How to Use
1. Upload CT DICOM series as a ZIP file
2. Upload MRI DICOM series as a ZIP file
3. Click **Run Registration**
4. Wait 2–5 minutes for processing
5. View previews and download results

## Stack
- Backend: FastAPI + SimpleITK + NumPy
- Frontend: Vanilla HTML/CSS/JS
- Deployment: Hugging Face Spaces (Docker)
