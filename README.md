# ct_mri_image_registration
Automatic CT–MRI image registration pipeline for medical imaging. Loads raw DICOM CT and MRI volumes, performs fully automatic 3D rigid registration using mutual information, resamples MRI to the CT grid, crops CT to the MRI field of view, and evaluates alignment using quantitative metrics (NMI, NCC, MAD, Edge Dice, FoV overlap).
