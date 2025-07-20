import numpy as np
from typing import Union, List, Tuple, Optional
from nilearn import image, input_data
from nibabel import Nifti1Image
from scipy.linalg import svd
from pathlib import Path
import pandas as pd


def create_ppi_regressor(voi, events_df, tr, microtime_resolution):


    pass

def create_raw_psych_regressor(
        events_df: pd.DataFrame,
        task_name: str,
        n_vols: int,
        tr: float,
        microtime_resolution: int = 16,
        amplitude_col: Optional[str] = None
) -> np.ndarray:
    """
    Creates a raw (unconvolved) psychological regressor from an events DataFrame.

    This function generates a vector with the same length as the number of scans,
    where the value at each time point corresponds to the state of the specified task.

    Args:
        events_df (pd.DataFrame):
            DataFrame with at least 'onset', 'duration', and 'trial_type' columns.
        task_name (str):
            The name of the condition in the 'trial_type' column to create the
            regressor for.
        n_vols (int):
            The total number of scans (time points) in the fMRI run.
        tr (float):
            The repetition time (TR) of the fMRI data in seconds.
        amplitude_col (str, optional):
            The name of a column in events_df to use for the amplitude of the
            regressor (for parametric modulation). If None, an amplitude of 1
            is used for all events of the specified task_name. Defaults to None.

    Returns:
        np.ndarray:
            A 1D numpy array of shape (n_scans,) representing the raw psychological
            regressor.
    """

    # Initialize an empty vector of zeros with length equal to the number of scans
    microtime_tr = tr / microtime_resolution
    total_microtime_bins = n_vols * microtime_resolution

    # Initialize the upsampled vector
    psych_vector_upsampled = np.zeros(total_microtime_bins)

    # Filter the DataFrame to get only the events for the specified task
    task_events = events_df[events_df['trial_type'] == task_name]

    if task_events.empty:
        print(f"Warning: No events found for trial_type '{task_name}'. Returning a zero vector.")
        return psych_vector_upsampled

    # Determine the amplitude for each event
    if amplitude_col:
        if amplitude_col not in task_events.columns:
            raise ValueError(f"Amplitude column '{amplitude_col}' not found in events DataFrame.")
        amplitudes = task_events[amplitude_col].values
        print(f"Using parametric modulation from column: '{amplitude_col}'")
    else:
        # If no amplitude column, all events have an amplitude of 1
        amplitudes = np.ones(len(task_events))

    # Iterate through each event and "turn on" the corresponding scans in the vector
    for i, (_, row) in enumerate(task_events.iterrows()):
        # Convert onset and duration from seconds to scan indices
        # np.round is important for accuracy
        start_scan = int(np.round(row['onset'] / microtime_tr))
        end_scan = int(np.round((row['onset'] + row['duration']) / microtime_tr))

        # Ensure indices are within the bounds of the scan length
        start_scan = max(0, start_scan)
        end_scan = min(n_vols, end_scan)

        # Assign the amplitude to the vector for the duration of the event
        psych_vector_upsampled[start_scan:end_scan] = amplitudes[i]
    return psych_vector_upsampled



def extract_vois_from_clean_img(
    clean_img: Union[str, Nifti1Image],
    seed_mask: Optional[Union[str, Nifti1Image, List[Union[str, Nifti1Image]]]] = None,
    seed_coords: Optional[Union[Tuple[float, float, float], List[Tuple[float, float, float]]]] = None,
    radius: float = 6.0,
    agg: str = 'eig'
) -> np.ndarray:
    """
    Extract time series from one or multiple regions of interest (ROIs) from a denoised fMRI image.

    This function extracts time series from ROIs defined by a binary mask, a list of binary masks,
    a brain atlas, or coordinates with a specified radius. The time series for each ROI is aggregated
    using either the mean or the first eigenvariate (SPM/gPPI style). The input image is assumed to be
    pre-denoised (e.g., confounds regressed out). The output is a NumPy array of shape (n_timepoints,)
    for a single ROI or (n_timepoints, n_rois) for multiple ROIs, with z-normalized time series.

    Parameters
    ----------
    clean_img : Union[str, Nifti1Image]
        Path to a 4D NIfTI file or a Nifti1Image object containing the denoised fMRI data.
    seed_mask : Optional[Union[str, Nifti1Image, List[Union[str, Nifti1Image]]]]
        A single binary mask (NIfTI file path or Nifti1Image), a list of binary masks, or a brain atlas
        (NIfTI file with integer labels for regions). If None, seed_coords must be provided.
    seed_coords : Optional[Union[Tuple[float, float, float], List[Tuple[float, float, float]]]]
        A single coordinate tuple (x, y, z) or a list of coordinate tuples in the same space as clean_img.
        Requires radius to define spherical ROIs. Ignored if seed_mask is provided.
    radius : float, optional
        Radius (in mm) for spherical ROIs when seed_coords is provided. Default is 6.0.
    agg : str, optional
        Aggregation method for ROI time series: 'mean' for averaging voxel time series or 'eig' for the
        first eigenvariate (SPM/gPPI style). Default is 'eig'.

    Returns
    -------
    np.ndarray
        Z-normalized time series array. Shape is (n_timepoints,) for a single ROI or
        (n_timepoints, n_rois) for multiple ROIs.

    Raises
    ------
    ValueError
        If neither seed_mask nor seed_coords is provided, if agg is invalid, or if input types are incorrect.
    """
    # Load clean_img
    clean_img = _load_nifti(clean_img)

    # Validate inputs
    _validate_inputs(clean_img, seed_mask, seed_coords, agg)

    # Initialize list to store time series
    time_series_list = []

    # Handle seed_mask (single mask, list of masks, or atlas)
    if seed_mask is not None:
        if isinstance(seed_mask, (str, Nifti1Image)):
            seed_mask = [seed_mask]  # Convert single mask to list
        elif not isinstance(seed_mask, list):
            raise ValueError("seed_mask must be a str, Nifti1Image, or list of str/Nifti1Image.")

        for mask in seed_mask:
            mask = _load_nifti(mask)
            # Check if mask is an atlas or binary mask
            mask_data = mask.get_fdata()
            if np.all(np.isin(mask_data, [0, 1])):  # Binary mask
                time_series = _extract_single_mask_time_series(clean_img, mask, agg)
                time_series_list.append(time_series)
            else:  # Atlas
                time_series_list.extend(_extract_atlas_time_series(clean_img, mask, agg))

    # Handle seed_coords
    elif seed_coords is not None:
        if isinstance(seed_coords, tuple):
            seed_coords = [seed_coords]  # Convert single coordinate to list
        elif not isinstance(seed_coords, list):
            raise ValueError("seed_coords must be a tuple or list of tuples.")
        time_series_list.extend(_extract_coords_time_series(clean_img, seed_coords, radius, agg))

    # Combine time series into array and z-normalize
    time_series_array = np.column_stack(time_series_list)
    time_series_array = (time_series_array - np.mean(time_series_array, axis=0)) / np.std(time_series_array, axis=0)

    # Return single array if only one ROI
    if time_series_array.shape[1] == 1:
        return time_series_array[:, 0]
    return time_series_array


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
        eigenvariate = u * np.sqrt(s[0]**2/ n)
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


def _extract_coords_time_series(
    clean_img: Nifti1Image,
    coords: List[Tuple[float, float, float]],
    radius: float,
    agg: str
) -> List[np.ndarray]:
    """
    Extract and aggregate time series from spherical ROIs defined by coordinates.

    Parameters
    ----------
    clean_img : Nifti1Image
        Denoised 4D fMRI image.
    coords : List[Tuple[float, float, float]]
        List of (x, y, z) coordinates for spherical ROIs.
    radius : float
        Radius of spherical ROIs (in mm).
    agg : str
        Aggregation method ('mean' or 'eig').

    Returns
    -------
    List[np.ndarray]
        List of aggregated time series, each of shape (n_timepoints,).
    """
    masker = input_data.NiftiSpheresMasker(coords, radius=radius)
    roi_time_series = masker.fit_transform(clean_img)
    time_series_list = []
    for i in range(roi_time_series.shape[1]):
        time_series = _aggregate_time_series(roi_time_series[:, [i]], agg)
        time_series_list.append(time_series)
    return time_series_list

def _extract_atlas_time_series(
    clean_img: Nifti1Image,
    atlas: Nifti1Image,
    agg: str
) -> List[np.ndarray]:
    """
    #TODO need to check
    Extract and aggregate time series from all regions in a brain atlas.

    Parameters
    ----------
    clean_img : Nifti1Image
        Denoised 4D fMRI image.
    atlas : Nifti1Image
        Atlas with integer labels for regions.
    agg : str
        Aggregation method ('mean' or 'eig').

    Returns
    -------
    List[np.ndarray]
        List of aggregated time series, each of shape (n_timepoints,).
    """
    atlas_data = atlas.get_fdata()
    labels = np.unique(atlas_data[atlas_data != 0]).astype(int)
    time_series_list = []
    for label in labels:
        label_mask = image.math_img(f"img == {label}", img=atlas)
        time_series = _extract_single_mask_time_series(clean_img, label_mask, agg)
        time_series_list.append(time_series)
    return time_series_list
