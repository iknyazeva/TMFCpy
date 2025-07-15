import numpy as np
from typing import Union, List, Tuple, Optional
from nilearn import image, input_data
from nibabel import Nifti1Image
from scipy.linalg import svd
from pathlib import Path




def extruct_vois_from_clean_img(clean_img, mask_img = None, seeds= None, radius = 6):
    pass


def _extract_single_mask_time_series(
    clean_img: Nifti1Image,
    mask: Nifti1Image,
    agg: str
) -> np.ndarray:
    """
    Extract and aggregate time series from a single binary mask.

    Parameters
    ----------
    clean_img : Nifti1Image
        Denoised 4D fMRI image.
    mask : Nifti1Image
        Binary mask for the ROI.
    agg : str
        Aggregation method ('mean' or 'eig').

    Returns
    -------
    np.ndarray
        Aggregated time series, shape (n_timepoints,).
    """
    masker = input_data.NiftiMasker(mask_img=mask)
    roi_time_series = masker.fit_transform(clean_img)
    return _aggregate_time_series(roi_time_series, agg)

def _aggregate_time_series(roi_time_series: np.ndarray, agg: str) -> np.ndarray:
    """
    Aggregate voxel time series within an ROI using mean or first eigenvariate (SPM/gPPI style).

    Parameters
    ----------
    roi_time_series : np.ndarray
        Time series of voxels in the ROI, shape (n_timepoints, n_voxels).
    agg : str
        Aggregation method: 'mean' or 'eig'.

    Returns
    -------
    np.ndarray
        Aggregated time series, shape (n_timepoints,).
    """
    if agg == 'mean':
        return np.mean(roi_time_series, axis=1)
    else:  # agg == 'eig'
        m, n = roi_time_series.shape  # m: time points, n: voxels
        # Center the data
        roi_time_series = roi_time_series - np.mean(roi_time_series, axis=0)
        # Compute SVD based on matrix dimensions
        if m > n:
            # SVD of covariance matrix (voxels x voxels)
            _, s, v = svd(roi_time_series.T @ roi_time_series, full_matrices=False)
            v = v[:, 0]  # First right singular vector
            u = roi_time_series @ v / np.sqrt(s[0])
        else:
            # SVD of time-time covariance matrix
            u, s, _ = svd(roi_time_series @ roi_time_series.T, full_matrices=False)
            u = u[:, 0]  # First left singular vector
            v = roi_time_series.T @ u / np.sqrt(s[0])
        # Sign adjustment based on voxel weights
        d = np.sign(np.sum(v))
        u = u * d
        v = v * d
        # Scale eigenvariate as in gPPI toolbox
        eigenvariate = u * np.sqrt(s[0] / n)
        return eigenvariate

def _validate_inputs(
    clean_img: Nifti1Image,
    seed_mask: Optional[Union[str, Nifti1Image, List[Union[str, Nifti1Image]]]],
    seed_coords: Optional[Union[Tuple[float, float, float], List[Tuple[float, float, float]]]],
    agg: str
) -> None:
    """
    Validate inputs for ROI extraction.

    Parameters
    ----------
    clean_img : Nifti1Image
        Denoised fMRI image.
    seed_mask : Optional[Union[str, Nifti1Image, List[Union[str, Nifti1Image]]]]
        Single mask, list of masks, or atlas.
    seed_coords : Optional[Union[Tuple[float, float, float], List[Tuple[float, float, float]]]]
        Single coordinate or list of coordinates.
    agg : str
        Aggregation method ('mean' or 'eig').

    Raises
    ------
    ValueError
        If inputs are invalid or inconsistent.
    """
    if seed_mask is None and seed_coords is None:
        raise ValueError("Either seed_mask or seed_coords must be provided.")
    if seed_mask is not None and seed_coords is not None:
        raise ValueError("Provide either seed_mask or seed_coords, not both.")
    if agg not in ['mean', 'eig']:
        raise ValueError("agg must be 'mean' or 'eig'.")
    if not isinstance(clean_img, Nifti1Image):
        raise ValueError("clean_img must be a Nifti1Image.")

def _load_nifti(img: Union[str, Nifti1Image, Path]) -> Nifti1Image:
    """
    Load a NIfTI image from a file path or return the Nifti1Image object.

    Parameters
    ----------
    img : Union[str, Nifti1Image, Path]
        Path to a NIfTI file or a Nifti1Image object.

    Returns
    -------
    Nifti1Image
        Loaded NIfTI image.

    Raises
    ------
    ValueError
        If the input is not a valid NIfTI file path or Nifti1Image object.
    """
    if isinstance(img, (str, Path)):
        return image.load_img(img)  # Uses nilearn.image.load_img, which returns nibabel.Nifti1Image
    if not isinstance(img, Nifti1Image):
        raise ValueError("Input must be a Nifti1Image or a valid file path.")
    return img

