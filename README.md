# ct_mri_image_registration
Automatic CT–MRI image registration pipeline for medical imaging. Loads raw DICOM CT and MRI volumes, performs fully automatic 3D rigid registration using mutual information, resamples MRI to the CT grid, crops CT to the MRI field of view, and evaluates alignment using quantitative metrics (NMI, NCC, MAD, Edge Dice, FoV overlap).
# CT–MRI Rigid Registration & Evaluation Pipeline

A complete **end-to-end medical image registration pipeline** for aligning **CT and MRI (DICOM)** volumes of the same patient, followed by **quantitative evaluation** and **visual quality checks**.  
Designed for **clinical R&D, validation, and research reporting**.

---

## 📌 What This Project Does (Straight Facts)

- Loads **raw CT & MRI DICOM series**
- Reorients both to a **common anatomical orientation (RAI)**
- Performs **fully automatic 3D rigid registration**
- Resamples **MRI → CT grid**
- Automatically detects **MRI field-of-view (FoV)** and crops CT
- Generates **bone mask** from CT (HU-based)
- Computes **industry-standard evaluation metrics**
- Produces **publication-ready visual previews**
- Saves **all intermediate and final outputs**

No manual landmarks. No GUI dependency. Fully reproducible.

---

## 🧠 Core Techniques Used

- **SimpleITK**
  - Mattes Mutual Information (MI)
  - VersorRigid3DTransform
  - Multi-resolution registration
  - Physical-space optimization
- **Automatic FoV detection**
- **CT HU-based bone masking**
- **Metric-driven validation**
- **Visual sanity checks (PNG)**

---

## 📂 Output Folder Structure

PatientXX/
├── raw_ct/ # (optional copy)
├── raw_mri/ # (optional copy)
├── registered/
│ ├── ct_fixed.nii.gz
│ ├── mri_original.nii.gz
│ ├── mri_rigid_to_ct.nii.gz
│ ├── ct_cropped_to_mri.nii.gz
│ ├── mri_rigid_cropped.nii.gz
│ ├── ct_mask_auto.nii.gz
│ ├── rigid_transform.tfm
│ ├── fov_bounds.json
│ └── metrics.json
└── previews/
├── ct_mid.png
├── mri_mid.png
├── overlay_mid.png
├── checkerboard_mid.png
├── edge_overlay_mid.png
└── organ_overlap_mid.png


---

## 📊 Evaluation Metrics Implemented

### Intensity-Based
- **NMI (Normalized Mutual Information)**
- **MAD (Mean Absolute Difference)**
- **NCC (Normalized Cross-Correlation)**

### Edge-Based
- **Edge Dice (3D Canny edges)**

### Spatial / Coverage
- **FoV Overlap Ratio**

### Optional (If Data Exists)
- **Dice / MSD / Hausdorff** (segmentation masks)
- **TRE** (landmark files)
- **Jacobian determinant check** (deformable fields)

All metrics are saved to `metrics.json`.

---

## 🖼️ Visual Validation (Why This Matters)

- **Overlay** → quick alignment sanity check  
- **Checkerboard** → local misalignment detection  
- **Edge overlay** → anatomical boundary agreement  
- **Bone mask overlay** → CT structural accuracy  

These catch failures that metrics alone won’t.

---

## ⚙️ How to Run

### 1️⃣ Requirements
```bash
pip install SimpleITK numpy matplotlib
```
Edit in script:

CT_RAW_DIR
MRI_RAW_DIR
BASE_OUTDIR
PATIENT_ID

Run
python main.py
