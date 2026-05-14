"""
pipeline.py — CT/MRI Registration Pipeline (refactored for web deployment)
All heavy processing lives here; app.py calls these functions.
"""

import os
import json
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  LOADING
# ─────────────────────────────────────────────

def load_series(series_dir: str, label: str) -> sitk.Image:
    """Load a DICOM series from a directory and return a SimpleITK Image."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_dir)
    if not series_ids:
        raise RuntimeError(f"[{label}] No DICOM series found in: {series_dir}")
    file_names = reader.GetGDCMSeriesFileNames(series_dir, series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()
    print(f"[{label}] Loaded {len(file_names)} slices.")
    return img


# ─────────────────────────────────────────────
#  PRE-PROCESSING
# ─────────────────────────────────────────────

def normalize(img: sitk.Image) -> sitk.Image:
    """Percentile-based [0,1] normalization."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mn, mx = np.percentile(arr, (1, 99))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0.0
    arr = np.clip(arr, 0, 1)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def reorient(img: sitk.Image) -> sitk.Image:
    """Reorient image to RAI so CT and MRI share the same direction."""
    return sitk.DICOMOrient(img, "RAI")


# ─────────────────────────────────────────────
#  REGISTRATION
# ─────────────────────────────────────────────

def register_rigid(
    ct: sitk.Image,
    mri: sitk.Image,
    num_iterations: int = 200,
    learning_rate: float = 1.0,
    num_histogram_bins: int = 50,
    sampling_percentage: float = 0.02,
) -> tuple[sitk.Image, sitk.Transform]:
    """
    Rigid 3-D registration: MRI → CT.

    Returns
    -------
    mri_rigid : sitk.Image   – MRI resampled onto the CT grid
    transform : sitk.Transform
    """
    ct_n  = normalize(ct)
    mri_n = normalize(mri)

    initial = sitk.CenteredTransformInitializer(
        ct_n,
        mri_n,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(num_histogram_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_percentage)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)

    final_transform = reg.Execute(ct_n, mri_n)
    print(f"[Registration] Final metric value: {reg.GetMetricValue():.6f}")

    mri_rigid = sitk.Resample(
        mri,
        ct,
        final_transform,
        sitk.sitkLinear,
        0.0,
        mri.GetPixelID(),
    )
    return mri_rigid, final_transform


# ─────────────────────────────────────────────
#  FIELD-OF-VIEW CROP
# ─────────────────────────────────────────────

def _mri_fov_bounds(mri_arr: np.ndarray, margin: int = 2) -> tuple:
    """Return (z_min, z_max, y_min, y_max, x_min, x_max) of non-zero MRI voxels."""
    mask = mri_arr != 0
    if not mask.any():
        raise RuntimeError("MRI rigid volume is all zero; FOV cannot be computed.")
    coords = np.array(np.nonzero(mask))
    z_min, y_min, x_min = coords.min(axis=1).tolist()
    z_max, y_max, x_max = coords.max(axis=1).tolist()

    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    z_max = min(z_max + margin, mri_arr.shape[0] - 1)
    y_max = min(y_max + margin, mri_arr.shape[1] - 1)
    x_max = min(x_max + margin, mri_arr.shape[2] - 1)

    return int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)


def crop_to_fov(
    ct: sitk.Image,
    mri_rigid: sitk.Image,
    margin: int = 2,
) -> tuple[sitk.Image, sitk.Image, dict]:
    """
    Crop both CT and registered MRI to the MRI field-of-view.

    Returns
    -------
    ct_crop   : sitk.Image
    mri_crop  : sitk.Image
    fov_info  : dict   – bounding-box coordinates
    """
    mri_arr = sitk.GetArrayFromImage(mri_rigid)
    z_min, z_max, y_min, y_max, x_min, x_max = _mri_fov_bounds(mri_arr, margin)

    size_z = z_max - z_min + 1
    size_y = y_max - y_min + 1
    size_x = x_max - x_min + 1

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([int(x_min), int(y_min), int(z_min)])
    roi.SetSize([int(size_x), int(size_y), int(size_z)])

    ct_crop  = roi.Execute(ct)
    mri_crop = roi.Execute(mri_rigid)

    fov_info = {
        "z_min": z_min, "z_max": z_max,
        "y_min": y_min, "y_max": y_max,
        "x_min": x_min, "x_max": x_max,
        "size_z": size_z, "size_y": size_y, "size_x": size_x,
    }
    return ct_crop, mri_crop, fov_info


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_nmi(fixed_img: sitk.Image, moving_img: sitk.Image, nbins: int = 64) -> float:
    fixed  = sitk.GetArrayFromImage(fixed_img).astype(np.float32).ravel()
    moving = sitk.GetArrayFromImage(moving_img).astype(np.float32).ravel()
    mask   = np.isfinite(fixed) & np.isfinite(moving)
    fixed, moving = fixed[mask], moving[mask]
    if fixed.size == 0:
        return 0.0
    hist_2d, _, _ = np.histogram2d(fixed, moving, bins=nbins)
    pxy = hist_2d / np.sum(hist_2d)
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)
    eps = 1e-12
    Hx  = -np.sum(px  * np.log(px  + eps))
    Hy  = -np.sum(py  * np.log(py  + eps))
    Hxy = -np.sum(pxy * np.log(pxy + eps))
    return float((Hx + Hy) / Hxy) if Hxy > 0 else 0.0


def compute_edge_dice(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    sigma: float = 1.0,
    lower: float = 0.1,
    upper: float = 0.3,
) -> float:
    def _norm01(a):
        a = a.copy().astype(np.float32)
        a -= a.min()
        mx = a.max()
        if mx > 0:
            a /= mx
        return a

    fixed_arr  = _norm01(sitk.GetArrayFromImage(fixed_img))
    moving_arr = _norm01(sitk.GetArrayFromImage(moving_img))

    def _to_sitk(arr, ref):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref)
        return img

    canny = sitk.CannyEdgeDetectionImageFilter()
    canny.SetVariance(sigma ** 2)
    canny.SetLowerThreshold(lower)
    canny.SetUpperThreshold(upper)

    fe = sitk.GetArrayFromImage(canny.Execute(_to_sitk(fixed_arr,  fixed_img)))  > 0
    me = sitk.GetArrayFromImage(canny.Execute(_to_sitk(moving_arr, moving_img))) > 0

    if fe.sum() == 0 or me.sum() == 0:
        return 0.0
    return float(2.0 * np.logical_and(fe, me).sum() / (fe.sum() + me.sum()))


def compute_mismatch_metrics(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
) -> tuple[float, float, float]:
    """Returns (MAD, NCC, 1-NCC)."""
    fixed  = sitk.GetArrayFromImage(fixed_img).astype(np.float32)
    moving = sitk.GetArrayFromImage(moving_img).astype(np.float32)
    mask   = np.isfinite(fixed) & np.isfinite(moving)
    fixed, moving = fixed[mask], moving[mask]
    if fixed.size == 0:
        return 0.0, 0.0, 0.0
    mad = float(np.mean(np.abs(fixed - moving)))
    fz  = fixed  - fixed.mean()
    mz  = moving - moving.mean()
    denom = np.linalg.norm(fz) * np.linalg.norm(mz)
    ncc   = float(np.dot(fz, mz) / denom) if denom > 0 else 0.0
    return mad, ncc, 1.0 - ncc


def compute_fov_overlap_ratio(ct_img: sitk.Image, fov_info: dict) -> float:
    total = np.prod(ct_img.GetSize())
    fov   = fov_info["size_x"] * fov_info["size_y"] * fov_info["size_z"]
    return float(fov) / float(total) if total > 0 else 0.0


def compute_all_metrics(
    ct: sitk.Image,
    ct_crop: sitk.Image,
    mri_crop: sitk.Image,
    fov_info: dict,
) -> dict:
    """Run all standard metrics and return a dict."""
    nmi          = compute_nmi(ct_crop, mri_crop)
    edge_dice    = compute_edge_dice(ct_crop, mri_crop)
    fov_ratio    = compute_fov_overlap_ratio(ct, fov_info)
    mad, ncc, mm = compute_mismatch_metrics(ct_crop, mri_crop)

    metrics = {
        "nmi_rigid_cropped":      round(nmi,       4),
        "edge_dice_rigid_cropped": round(edge_dice, 4),
        "fov_overlap_ratio":      round(fov_ratio,  4),
        "mad_rigid_cropped":      round(mad,        4),
        "ncc_rigid_cropped":      round(ncc,        4),
        "mismatch_rigid_cropped": round(mm,         4),
    }

    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# ─────────────────────────────────────────────
#  BONE MASK  (for preview overlay)
# ─────────────────────────────────────────────

def create_ct_bone_mask(ct_img: sitk.Image, hu_threshold: float = 150) -> sitk.Image:
    arr  = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask = sitk.GetImageFromArray((arr > hu_threshold).astype(np.uint8))
    mask.CopyInformation(ct_img)
    return mask


# ─────────────────────────────────────────────
#  PREVIEW IMAGE GENERATION
# ─────────────────────────────────────────────

def _mid_slice_arr(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)
    mid = arr.shape[0] // 2
    return arr[mid].astype(np.float32)


def _nrm(a: np.ndarray) -> np.ndarray:
    a = a.copy().astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    return a


def _checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    h, w = a.shape
    out  = np.zeros_like(a, dtype=np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            if ((y // tile) + (x // tile)) % 2 == 0:
                out[y:y2, x:x2] = a[y:y2, x:x2]
            else:
                out[y:y2, x:x2] = b[y:y2, x:x2]
    return out


def _mask_overlay(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b   = _nrm(base)
    rgb = np.stack([b, b, b], axis=-1)
    rgb[mask.astype(bool), 0] = 1.0
    rgb[mask.astype(bool), 1] = 0.0
    rgb[mask.astype(bool), 2] = 0.0
    return rgb


def _edge_overlay(ct: np.ndarray, mri: np.ndarray, thresh: float = 0.2) -> np.ndarray:
    ct  = _nrm(ct)
    mr  = _nrm(mri)
    gx_ct, gy_ct = np.gradient(ct)
    gx_mr, gy_mr = np.gradient(mr)
    ct_edges = np.sqrt(gx_ct**2 + gy_ct**2) > (thresh * np.sqrt(gx_ct**2 + gy_ct**2).max())
    mr_edges = np.sqrt(gx_mr**2 + gy_mr**2) > (thresh * np.sqrt(gx_mr**2 + gy_mr**2).max())
    rgb = np.stack([ct, ct, ct], axis=-1)
    rgb[ct_edges, :] = [1.0, 0.0, 0.0]
    rgb[mr_edges, 0] = 1.0
    rgb[mr_edges, 1] = 1.0
    rgb[mr_edges, 2] = 0.0
    return rgb


def save_previews(
    ct_crop: sitk.Image,
    mri_crop: sitk.Image,
    out_dir: str,
) -> list[str]:
    """
    Save 6 PNG preview images to out_dir.
    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    ct_np  = _mid_slice_arr(ct_crop)
    mri_np = _mid_slice_arr(mri_crop)
    ct_d   = _nrm(ct_np)
    mri_d  = _nrm(mri_np)

    mask_img  = create_ct_bone_mask(ct_crop)
    mask_np   = _mid_slice_arr(mask_img)

    saved = []
    def _save(name, arr, cmap="gray"):
        p = os.path.join(out_dir, name)
        plt.imsave(p, arr, cmap=cmap)
        saved.append(p)
        return p

    _save("ct_mid.png",             ct_d)
    _save("mri_mid.png",            mri_d)
    _save("overlay_mid.png",        0.5 * ct_d + 0.5 * mri_d)
    _save("checkerboard_mid.png",   _checkerboard(ct_d, mri_d))
    _save("edge_overlay_mid.png",   _edge_overlay(ct_np, mri_np), cmap=None)
    _save("organ_overlap_mid.png",  _mask_overlay(0.5*ct_d + 0.5*mri_d, mask_np), cmap=None)

    print(f"[Previews] Saved {len(saved)} images to {out_dir}")
    return saved


# ─────────────────────────────────────────────
#  FULL PIPELINE  (called by app.py)
# ─────────────────────────────────────────────

def run_full_pipeline(ct_dir: str, mri_dir: str, out_dir: str) -> dict:
    """
    End-to-end pipeline.

    Parameters
    ----------
    ct_dir  : folder containing CT DICOM files
    mri_dir : folder containing MRI DICOM files
    out_dir : root output folder for this job

    Returns
    -------
    dict with keys: metrics, output_files
    """
    reg_dir  = os.path.join(out_dir, "registered")
    prev_dir = os.path.join(out_dir, "previews")
    os.makedirs(reg_dir,  exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    # 1. Load
    print("[Pipeline] Loading CT...")
    ct  = load_series(ct_dir,  "CT")
    print("[Pipeline] Loading MRI...")
    mri = load_series(mri_dir, "MRI")

    # 2. Reorient
    ct  = reorient(ct)
    mri = reorient(mri)

    # 3. Register
    print("[Pipeline] Registering...")
    mri_rigid, transform = register_rigid(ct, mri)

    # 4. Crop to FOV
    print("[Pipeline] Cropping to MRI FOV...")
    ct_crop, mri_crop, fov_info = crop_to_fov(ct, mri_rigid)

    # 5. Bone mask
    ct_mask = create_ct_bone_mask(ct_crop)

    # 6. Save NIfTI volumes
    output_files = {}
    files_to_save = {
        "ct_fixed.nii.gz":          ct,
        "mri_original.nii.gz":      mri,
        "mri_rigid_to_ct.nii.gz":   mri_rigid,
        "ct_cropped.nii.gz":        ct_crop,
        "mri_cropped.nii.gz":       mri_crop,
        "ct_mask_auto.nii.gz":      ct_mask,
    }
    for fname, img in files_to_save.items():
        p = os.path.join(reg_dir, fname)
        sitk.WriteImage(img, p)
        output_files[fname] = p

    # Save transform
    tfm_path = os.path.join(reg_dir, "rigid_transform.tfm")
    sitk.WriteTransform(transform, tfm_path)
    output_files["rigid_transform.tfm"] = tfm_path

    # Save FOV JSON
    fov_path = os.path.join(reg_dir, "fov_bounds.json")
    with open(fov_path, "w") as f:
        json.dump(fov_info, f, indent=4)
    output_files["fov_bounds.json"] = fov_path

    # 7. Metrics
    print("[Pipeline] Computing metrics...")
    metrics = compute_all_metrics(ct, ct_crop, mri_crop, fov_info)
    metrics_path = os.path.join(reg_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    output_files["metrics.json"] = metrics_path

    # 8. Previews
    print("[Pipeline] Generating previews...")
    preview_paths = save_previews(ct_crop, mri_crop, prev_dir)
    for p in preview_paths:
        output_files[os.path.basename(p)] = p

    print("[Pipeline] DONE.")
    return {"metrics": metrics, "output_files": output_files}