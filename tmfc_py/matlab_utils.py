import numpy as np
from nilearn import image
import glob
import os
import re
import nibabel as nib
from typing import List, Union, Tuple, Optional
import scipy.io as sio
import pandas as pd


def create_events_from_mat(file_path: str) -> pd.DataFrame:
    """Create a sorted pandas DataFrame of events (for nilearn) from a MATLAB file.

    This function loads a MATLAB file containing event data (onsets, durations, names),
    processes the arrays, and returns a DataFrame with columns 'trial_type', 'onset',
    and 'duration', sorted by onset time. Negative onsets are filtered out.

    :param file_path: Path to the MATLAB file (.mat).
    :type file_path: str
    :return: DataFrame with columns 'trial_type', 'onset', and 'duration'.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the specified file does not exist.
    :raises KeyError: If the expected keys ('onsets', 'durations', 'names') are missing in the .mat file.
    """

    # Load MATLAB file
    mat_struct = sio.loadmat(file_path)

    # Extract and squeeze arrays to remove unnecessary dimensions
    onsets: np.ndarray = mat_struct['onsets'].squeeze()
    durations: np.ndarray = mat_struct['durations'].squeeze()
    names: np.ndarray = mat_struct['names'].squeeze()

    # Initialize lists to store flattened data
    onset_list: List[float] = []
    name_list: List[str] = []
    duration_list: List[float] = []
    # Process each event
    for onset, duration, name in zip(onsets, durations, names):
        onset, duration, name = onset.squeeze(), duration.squeeze(), name.squeeze()
        num_events: int = len(onset)

        # Replicate name and duration if duration is a single value
        name_list.extend([name] * num_events)
        if duration.shape == ():
            duration = [duration] * num_events
        duration_list.extend(duration)
        onset_list.extend(onset)

        # Create and sort DataFrame by onset
    events_df = pd.DataFrame({
        "trial_type": name_list,
        "onset": onset_list,
        "duration": duration_list
    }).sort_values(by="onset")

    # Filter out negative onsets
    events_df = events_df[events_df["onset"] >= 0]
    return events_df

def convert_3d_to_4d_fmri(
    directory_path: str,
    file_pattern: Optional[str] = None,
    auto_resample: bool = False
) -> nib.Nifti1Image:
    """
    Convert a sequence of 3D fMRI NIfTI files into a single 4D NIfTI image.

    This function searches for 3D NIfTI files in the specified directory matching
    the given pattern, sorts them by volume number extracted from filenames, and
    concatenates them into a 4D NIfTI image using nilearn's `concat_imgs`.

    Args:
        directory_path (str): Path to the directory containing 3D NIfTI files.
        file_pattern (Optional[str]): Glob pattern to match NIfTI files.
            Defaults to "swarsSST_*.nii" if not provided.
        auto_resample (bool): If True, automatically resample images to match
            the affine of the first image during concatenation. Defaults to False.

    Returns:
        nib.Nifti1Image: A 4D NIfTI image combining all input 3D images.

    Raises:
        FileNotFoundError: If no files matching the pattern are found in the directory.
        ValueError: If a filename does not contain a valid volume number.
    """
    if file_pattern is None:
        file_pattern = "swarsSST_*.nii"

    # Construct the full path pattern for glob
    full_path_pattern = os.path.join(directory_path, file_pattern)
    nii_files = glob.glob(full_path_pattern)

    if not nii_files:
        raise FileNotFoundError(
            f"No NIfTI files found in {directory_path} matching pattern '{file_pattern}'"
        )

    def _extract_volume_number(filename: str) -> int:
        """
        Extract the volume number from a NIfTI filename.

        Args:
            filename (str): The full path or basename of a NIfTI file.

        Returns:
            int: The extracted volume number.

        Raises:
            ValueError: If the volume number cannot be extracted from the filename.
        """
        match = re.search(r'(\d{6}-\d{2})\.nii$', os.path.basename(filename))
        if not match:
            raise ValueError(f"Cannot extract volume number from filename: {filename}")
        return int(match.group(1).split('-')[0])

    # Sort files by volume number
    sorted_nii_files = sorted(nii_files, key=_extract_volume_number)

    # Load 3D NIfTI images into a list
    nii_images: List[nib.Nifti1Image] = [
        image.load_img(filename) for filename in sorted_nii_files
    ]

    # Concatenate 3D images into a 4D image
    fmri_4d: nib.Nifti1Image = image.concat_imgs(
        nii_images, auto_resample=auto_resample
    )

    return fmri_4d