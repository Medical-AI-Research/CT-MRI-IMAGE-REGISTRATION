import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ===========================================================
#                USER SETTINGS 
# ============================================================

PATIENT_ID   = "Patient11"

# Base folder where pipeline will save everything
BASE_OUTDIR  = r"D:\medical_image_registration"

# Raw CT + MRI locations (input folders with DICOM files)
CT_RAW_DIR   = r"D:\other datasets\saroja\CT\173199\raw_ct"
MRI_RAW_DIR  = r"D:\other datasets\saroja\MRI\1\mri_transverse"

# If True → copies raw DICOMs into PatientXX/raw_ct, raw_mri
COPY_RAW_DICOM = False

# ============================================================


def make_patient_dirs():
    patient_dir = os.path.join(BASE_OUTDIR, PATIENT_ID)
    paths = {
        "patient":   patient_dir,
        "raw_ct":    os.path.join(patient_dir, "raw_ct"),
        "raw_mri":   os.path.join(patient_dir, "raw_mri"),
        "registered": os.path.join(patient_dir, "registered"),
        "previews":  os.path.join(patient_dir, "previews"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def copy_raw_if_needed(paths):
    if not COPY_RAW_DICOM:
        return
    for src, dst in [(CT_RAW_DIR, paths["raw_ct"]), (MRI_RAW_DIR, paths["raw_mri"])]:
        for f in os.listdir(src):
            src_f = os.path.join(src, f)
            if os.path.isfile(src_f):
                shutil.copy(src_f, dst)
    print("Raw DICOMs copied.")


def load_series(series_dir, label):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_dir)
    if not series_ids:
        raise RuntimeError(f"[{label}] No DICOM series in {series_dir}")
    file_names = reader.GetGDCMSeriesFileNames(series_dir, series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()
    print(f"[{label}] Loaded {len(file_names)} slices.")
    return img


def normalize(img):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mn, mx = np.percentile(arr, (1, 99))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0
    arr = np.clip(arr, 0, 1)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def mid_slice(img):
    size = list(img.GetSize())
    index = [0, 0, size[2] // 2]
    size[2] = 0
    ex = sitk.ExtractImageFilter()
    ex.SetSize(size)
    ex.SetIndex(index)
    out = ex.Execute(img)
    return out


def nrm(a):
    a = a.astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    return a

def make_checkerboard(slice_a, slice_b, tile_size=100):
    """
    Create a 2D checkerboard image from two slices (same shape).
    Alternates tiles between slice_a and slice_b.
    """
    if slice_a.shape != slice_b.shape:
        raise ValueError("Checkerboard inputs must have the same shape.")

    h, w = slice_a.shape
    out = np.zeros_like(slice_a, dtype=np.float32)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)

            # Decide which image to use for this tile
            use_a = ((y // tile_size) + (x // tile_size)) % 2 == 0
            if use_a:
                out[y:y2, x:x2] = slice_a[y:y2, x:x2]
            else:
                out[y:y2, x:x2] = slice_b[y:y2, x:x2]

    return out
def create_ct_bone_mask(ct_img, hu_threshold=150):
    """
    Rough bone mask from CT using a HU threshold.
    ct_img: original CT (not normalized).
    """
    arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask = arr > hu_threshold  # bone-ish

    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_img.CopyInformation(ct_img)
    return mask_img


def make_mask_overlay(base_slice, mask_slice):
    """
    RGB overlay: base_slice in gray, mask in red.
    base_slice, mask_slice: 2D numpy arrays.
    """
    base = nrm(base_slice.astype(np.float32))
    mask = mask_slice.astype(bool)

    rgb = np.stack([base, base, base], axis=-1)
    rgb[mask, 0] = 1.0  # R
    rgb[mask, 1] = 0.0  # G
    rgb[mask, 2] = 0.0  # B
    return rgb


def make_edge_overlay(ct_slice, mri_slice, edge_thresh=0.2):
    """
    RGB overlay: CT as gray, CT edges in red, MRI edges in green.
    ct_slice, mri_slice: 2D numpy arrays (same shape).
    """
    if ct_slice.shape != mri_slice.shape:
        raise ValueError("CT and MRI slices must have same shape for edge overlay.")

    ct = nrm(ct_slice.astype(np.float32))
    mr = nrm(mri_slice.astype(np.float32))

    # simple gradient-based edges
    gx_ct, gy_ct = np.gradient(ct)
    gx_mr, gy_mr = np.gradient(mr)

    mag_ct = np.sqrt(gx_ct**2 + gy_ct**2)
    mag_mr = np.sqrt(gx_mr**2 + gy_mr**2)

    ct_edges = mag_ct > (edge_thresh * mag_ct.max())
    mr_edges = mag_mr > (edge_thresh * mag_mr.max())

    rgb = np.stack([ct, ct, ct], axis=-1)

    # CT edges → red
    rgb[ct_edges, 0] = 1.0
    rgb[ct_edges, 1] = 0.0
    rgb[ct_edges, 2] = 0.0

    # MRI edges → green (if both, becomes yellow)
    rgb[mr_edges, 0] = 1.0
    rgb[mr_edges, 1] = 1.0
    rgb[mr_edges, 2] = 0.0

    return rgb

def compute_nmi(fixed_img, moving_img, nbins: int = 64) -> float:
    """NMI between two images on same grid."""
    fixed_arr = sitk.GetArrayFromImage(fixed_img).astype(np.float32).ravel()
    moving_arr = sitk.GetArrayFromImage(moving_img).astype(np.float32).ravel()

    mask = np.logical_and(np.isfinite(fixed_arr), np.isfinite(moving_arr))
    fixed_arr = fixed_arr[mask]
    moving_arr = moving_arr[mask]
    if fixed_arr.size == 0:
        return 0.0

    hist_2d, _, _ = np.histogram2d(fixed_arr, moving_arr, bins=nbins)
    pxy = hist_2d / np.sum(hist_2d)

    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    eps = 1e-12
    Hx = -np.sum(px * np.log(px + eps))
    Hy = -np.sum(py * np.log(py + eps))
    Hxy = -np.sum(pxy * np.log(pxy + eps))

    if Hxy <= 0:
        return 0.0

    nmi = (Hx + Hy) / Hxy
    return float(nmi)


def compute_edge_dice(fixed_img, moving_img, sigma=1.0, lower=0.1, upper=0.3) -> float:
    """Edge-based Dice using 3D Canny edges."""
    fixed_arr = sitk.GetArrayFromImage(fixed_img).astype(np.float32)
    moving_arr = sitk.GetArrayFromImage(moving_img).astype(np.float32)

    def norm01(a):
        a = a.copy()
        a -= a.min()
        maxv = a.max()
        if maxv > 0:
            a /= maxv
        return a

    fixed_arr = norm01(fixed_arr)
    moving_arr = norm01(moving_arr)

    fixed_n = sitk.GetImageFromArray(fixed_arr)
    moving_n = sitk.GetImageFromArray(moving_arr)
    fixed_n.CopyInformation(fixed_img)
    moving_n.CopyInformation(moving_img)

    canny = sitk.CannyEdgeDetectionImageFilter()
    canny.SetVariance(sigma**2)
    canny.SetLowerThreshold(lower)
    canny.SetUpperThreshold(upper)

    fixed_edges = canny.Execute(fixed_n)
    moving_edges = canny.Execute(moving_n)

    fe = sitk.GetArrayFromImage(fixed_edges) > 0
    me = sitk.GetArrayFromImage(moving_edges) > 0

    fe_sum = fe.sum()
    me_sum = me.sum()
    if fe_sum == 0 or me_sum == 0:
        return 0.0

    intersection = np.logical_and(fe, me).sum()
    dice = 2.0 * intersection / (fe_sum + me_sum)
    return float(dice)


def compute_mismatch_metrics(fixed_img, moving_img):
    """Average mismatch: MAD, NCC, and 1-NCC."""
    fixed = sitk.GetArrayFromImage(fixed_img).astype(np.float32)
    moving = sitk.GetArrayFromImage(moving_img).astype(np.float32)

    mask = np.logical_and(np.isfinite(fixed), np.isfinite(moving))
    fixed = fixed[mask]
    moving = moving[mask]
    if fixed.size == 0:
        return 0.0, 0.0, 0.0

    mad = float(np.mean(np.abs(fixed - moving)))

    fixed_z = fixed - fixed.mean()
    moving_z = moving - moving.mean()
    denom = np.linalg.norm(fixed_z) * np.linalg.norm(moving_z)
    if denom == 0:
        ncc = 0.0
    else:
        ncc = float(np.dot(fixed_z, moving_z) / denom)

    mismatch = 1.0 - ncc
    return mad, ncc, mismatch


def compute_region_metrics(fixed_mask, moving_mask):
    """
    Dice, MSD (average Hausdorff), and Hausdorff distance between two binary masks.
    Only works if you provide segmentation masks.
    """
    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(fixed_mask, moving_mask)
    dice = float(overlap.GetDiceCoefficient())

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(fixed_mask, moving_mask)
    msd = float(hd_filter.GetAverageHausdorffDistance())
    hausdorff = float(hd_filter.GetHausdorffDistance())
    return dice, msd, hausdorff


def compute_tre(fixed_points, moving_points, spacing=None):
    """
    TRE between corresponding landmark points.
    fixed_points / moving_points: list of [x,y,z] in index space.
    spacing: optional voxel spacing to convert to mm.
    """
    fixed_pts = np.asarray(fixed_points, dtype=np.float32)
    moving_pts = np.asarray(moving_points, dtype=np.float32)
    if fixed_pts.shape != moving_pts.shape:
        raise ValueError("fixed_points and moving_points must have same shape")
    diff = fixed_pts - moving_pts
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=np.float32)
        diff = diff * spacing
    dists = np.linalg.norm(diff, axis=1)
    return float(np.mean(dists)), float(np.std(dists))


def check_jacobian_determinant(displacement_field):
    """
    Jacobian determinant summary for a displacement field.
    Requires a deformable displacement field image.
    """
    jac = sitk.DisplacementFieldJacobianDeterminant(displacement_field)
    arr = sitk.GetArrayFromImage(jac)
    min_det = float(arr.min())
    num_neg = int((arr < 0).sum())
    return min_det, num_neg


def compute_fov_overlap_ratio(ct_img, size_x, size_y, size_z):
    """FOV overlap ratio: cropped CT volume / full CT volume."""
    total_vox = np.prod(ct_img.GetSize())
    fov_vox = size_x * size_y * size_z
    if total_vox == 0:
        return 0.0
    return float(fov_vox) / float(total_vox)


def compute_mri_fov_bounds(mri_rigid_arr, margin=2):
    """
    Compute the 3D bounding box (z,y,x) of non-zero MRI region.
    Returns (z_min, z_max, y_min, y_max, x_min, x_max).
    """
    mask = mri_rigid_arr != 0
    if not mask.any():
        raise RuntimeError("MRI rigid volume is all zero; FOV cannot be computed.")

    coords = np.array(np.nonzero(mask))
    z_min, y_min, x_min = coords.min(axis=1)
    z_max, y_max, x_max = coords.max(axis=1)

    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    z_max = min(z_max + margin, mri_rigid_arr.shape[0] - 1)
    y_max = min(y_max + margin, mri_rigid_arr.shape[1] - 1)
    x_max = min(x_max + margin, mri_rigid_arr.shape[2] - 1)

    return int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)


def main():
    # ------------ CREATE PATIENT FOLDERS -------------
    paths = make_patient_dirs()
    copy_raw_if_needed(paths)

    # ------------ LOAD RAW CT & MRI -------------
    print("\nLoading CT...")
    ct = load_series(CT_RAW_DIR, "CT")

    print("Loading MRI...")
    mri = load_series(MRI_RAW_DIR, "MRI")

    # Reorient both to same coordinate direction
    ct = sitk.DICOMOrient(ct, "RAI")
    mri = sitk.DICOMOrient(mri, "RAI")

    # ------------ NORMALIZE FOR REGISTRATION -------------
    ct_n  = normalize(ct)
    mri_n = normalize(mri)

    # ------------ RIGID 3D REGISTRATION -------------
    print("\nPerforming 3D rigid registration...")

    initial = sitk.CenteredTransformInitializer(
        ct_n,
        mri_n,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)

    final_rigid = reg.Execute(ct_n, mri_n)
    print("Rigid registration complete.")
    print("Metric:", reg.GetMetricValue())

    # ------------ RESAMPLE MRI TO CT GRID -------------
    mri_rigid = sitk.Resample(
        mri,
        ct,
        final_rigid,
        sitk.sitkLinear,
        0.0,
        mri.GetPixelID(),
    )

    reg_dir = paths["registered"]

    ct_fixed_path   = os.path.join(reg_dir, "ct_fixed.nii.gz")
    mri_orig_path   = os.path.join(reg_dir, "mri_original.nii.gz")
    mri_rigid_path  = os.path.join(reg_dir, "mri_rigid_to_ct.nii.gz")
    rigid_tfm_path  = os.path.join(reg_dir, "rigid_transform.tfm")

    sitk.WriteImage(ct, ct_fixed_path)
    sitk.WriteImage(mri, mri_orig_path)
    sitk.WriteImage(mri_rigid, mri_rigid_path)
    sitk.WriteTransform(final_rigid, rigid_tfm_path)

    # ------------ CROP CT TO MRI'S FIELD OF VIEW -------------
    print("\n[FoV] Computing MRI field-of-view and cropping CT...")

    mri_arr = sitk.GetArrayFromImage(mri_rigid)  # (z, y, x)
    z_min, z_max, y_min, y_max, x_min, x_max = compute_mri_fov_bounds(mri_arr, margin=2)

    size_z = z_max - z_min + 1
    size_y = y_max - y_min + 1
    size_x = x_max - x_min + 1

    fov_info = {
        "z_min": z_min, "z_max": z_max,
        "y_min": y_min, "y_max": y_max,
        "x_min": x_min, "x_max": x_max,
        "size_z": size_z, "size_y": size_y, "size_x": size_x
    }

    fov_json_path = os.path.join(reg_dir, "fov_bounds.json")
    with open(fov_json_path, "w") as f:
        json.dump(fov_info, f, indent=4)

    print("[FoV] Bounds (z):", z_min, "->", z_max)
    print("[FoV] Bounds (y):", y_min, "->", y_max)
    print("[FoV] Bounds (x):", x_min, "->", x_max)
    print("[FoV] Saved FOV info to:", fov_json_path)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([int(x_min), int(y_min), int(z_min)])  # x, y, z
    roi.SetSize([int(size_x), int(size_y), int(size_z)])

    ct_crop  = roi.Execute(ct)
    mri_crop = roi.Execute(mri_rigid)

    ct_crop_path  = os.path.join(reg_dir, "ct_cropped_to_mri.nii.gz")
    mri_crop_path = os.path.join(reg_dir, "mri_rigid_cropped.nii.gz")

        # Auto CT bone mask for organ overlap visualization
    ct_mask = create_ct_bone_mask(ct_crop, hu_threshold=150)
    ct_mask_path = os.path.join(reg_dir, "ct_mask_auto.nii.gz")
    
    sitk.WriteImage(ct_mask, ct_mask_path)
    sitk.WriteImage(ct_crop,  ct_crop_path)
    sitk.WriteImage(mri_crop, mri_crop_path)

    print("Cropped volumes saved:")
    print("   ", ct_crop_path)
    print("   ", mri_crop_path)

    # ------------ METRICS (RIGID, CROPPED FOV) -------------
    metrics = {}

    # 1. NMI
    nmi_rigid_cropped = compute_nmi(ct_crop, mri_crop)
    metrics["nmi_rigid_cropped"] = nmi_rigid_cropped

    # 2. Edge Alignment (Edge Dice)
    edge_dice_rigid_cropped = compute_edge_dice(ct_crop, mri_crop)
    metrics["edge_dice_rigid_cropped"] = edge_dice_rigid_cropped

    # 3. FOV Overlap Ratio
    fov_overlap_ratio = compute_fov_overlap_ratio(ct, size_x, size_y, size_z)
    metrics["fov_overlap_ratio"] = fov_overlap_ratio

    # 4. Average mismatch (MAD, NCC, 1-NCC)
    mad, ncc, mismatch = compute_mismatch_metrics(ct_crop, mri_crop)
    metrics["mad_rigid_cropped"] = mad
    metrics["ncc_rigid_cropped"] = ncc
    metrics["mismatch_rigid_cropped"] = mismatch

    print(f"NMI (rigid, cropped FOV):             {nmi_rigid_cropped:.4f}")
    print(f"Edge Dice (rigid, cropped FOV):       {edge_dice_rigid_cropped:.4f}")
    print(f"FOV overlap ratio (cropped / full):   {fov_overlap_ratio:.4f}")
    print(f"MAD (rigid, cropped FOV):             {mad:.4f}")
    print(f"NCC (rigid, cropped FOV):             {ncc:.4f}")
    print(f"Average mismatch (1 - NCC):           {mismatch:.4f}")

    # 5–7. Optional region metrics (Dice, MSD, Hausdorff) if masks are available
    ct_mask_path  = os.path.join(reg_dir, "ct_mask.nii.gz")
    mri_mask_path = os.path.join(reg_dir, "mri_mask.nii.gz")
    if os.path.exists(ct_mask_path) and os.path.exists(mri_mask_path):
        print("\nFound masks, computing region metrics (Dice/MSD/Hausdorff)...")
        ct_mask  = sitk.ReadImage(ct_mask_path)
        mri_mask = sitk.ReadImage(mri_mask_path)
        dice, msd, hausdorff = compute_region_metrics(ct_mask, mri_mask)
        metrics["dice_region"] = dice
        metrics["msd_region"] = msd
        metrics["hausdorff_region"] = hausdorff
        print(f"Dice (region masks):                  {dice:.4f}")
        print(f"MSD / avg Hausdorff (region masks):   {msd:.4f}")
        print(f"Hausdorff distance (region masks):    {hausdorff:.4f}")
    else:
        print("\nNo region masks found -> Dice/MSD/Hausdorff not computed.")

    # 8. Optional TRE if landmark files exist
    tre_fixed_path  = os.path.join(reg_dir, "landmarks_fixed.json")
    tre_moving_path = os.path.join(reg_dir, "landmarks_moving.json")
    if os.path.exists(tre_fixed_path) and os.path.exists(tre_moving_path):
        print("Found landmark files, computing TRE...")
        with open(tre_fixed_path, "r") as f:
            fixed_pts = json.load(f)
        with open(tre_moving_path, "r") as f:
            moving_pts = json.load(f)
        tre_mean, tre_std = compute_tre(fixed_pts, moving_pts, spacing=ct_crop.GetSpacing())
        metrics["tre_mean_mm"] = tre_mean
        metrics["tre_std_mm"] = tre_std
        print(f"TRE mean (mm):                        {tre_mean:.4f}")
        print(f"TRE std  (mm):                        {tre_std:.4f}")
    else:
        print("No landmark files found -> TRE not computed.")

    # 9. Optional Jacobian check if deformable displacement field exists
    disp_field_path = os.path.join(reg_dir, "deformable_displacement.nii.gz")
    if os.path.exists(disp_field_path):
        print("Found displacement field, checking Jacobian determinant...")
        disp_field = sitk.ReadImage(disp_field_path, sitk.sitkVectorFloat64)
        min_det, num_neg = check_jacobian_determinant(disp_field)
        metrics["jacobian_min_det"] = min_det
        metrics["jacobian_num_negative"] = num_neg
        print(f"Jacobian min determinant:             {min_det:.4f}")
        print(f"Jacobian negative-voxel count:        {num_neg}")
    else:
        print("No deformable field -> Jacobian not computed.")

    metrics_path = os.path.join(reg_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to:", metrics_path)

    # ------------ PREVIEW PNGs -------------
    print("\nGenerating preview images...")
    prev_dir = paths["previews"]

    ct_mid   = mid_slice(ct_crop)
    mri_mid  = mid_slice(mri_crop)
    mask_mid = mid_slice(ct_mask)

    ct_np   = np.squeeze(sitk.GetArrayFromImage(ct_mid))
    mri_np  = np.squeeze(sitk.GetArrayFromImage(mri_mid))
    mask_np = np.squeeze(sitk.GetArrayFromImage(mask_mid))

    ct_d  = nrm(ct_np)
    mri_d = nrm(mri_np)
    overlay = 0.5 * ct_d + 0.5 * mri_d

    # basic grayscale previews
    plt.imsave(os.path.join(prev_dir, "ct_mid.png"), ct_d, cmap="gray")
    plt.imsave(os.path.join(prev_dir, "mri_mid.png"), mri_d, cmap="gray")
    plt.imsave(os.path.join(prev_dir, "overlay_mid.png"), overlay, cmap="gray")

    # (4) Organ mask overlay (auto bone mask on CT+MRI overlay)
    organ_overlay = make_mask_overlay(overlay, mask_np)
    plt.imsave(os.path.join(prev_dir, "organ_overlap_mid.png"), organ_overlay)

    # (3) Edge overlay: CT edges + MRI edges
    edge_overlay = make_edge_overlay(ct_np, mri_np, edge_thresh=0.2)
    plt.imsave(os.path.join(prev_dir, "edge_overlay_mid.png"), edge_overlay)

    # Checkerboard visualization (CT vs registered MRI)
    checker = make_checkerboard(ct_d, mri_d, tile_size=32)
    plt.imsave(os.path.join(prev_dir, "checkerboard_mid.png"), checker, cmap="gray")

    print("\nPIPELINE COMPLETE FOR:", PATIENT_ID)



if __name__ == "__main__":
    main()
