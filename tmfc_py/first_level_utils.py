import numpy as np
import glob
import pandas as pd
import os
from nilearn import image
from typing import Literal, Optional, List
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel


def create_motion_confounds_from_rp(
        rp_dir: str,
        n_params: Literal[6, 12, 24] = 6
) -> pd.DataFrame:

    """
    Load motion parameters from a text file and expand to 6, 12, or 24 regressors.

    :param filepath:
        Path to the text file containing the 6 motion parameters.
    :type filepath: str
    :param n_params:
        Desired number of motion regressors. Must be one of ``6``, ``12``, or ``24``.
        - ``6`` returns the original 6 motion parameters.
        - ``12`` returns the original 6 plus their temporal derivatives.
        - ``24`` returns the original 12 regressors plus their squared terms.
    :type n_params: Literal[6, 12, 24]
    :param sep:
        Delimiter used in the text file (default is tab ``\\t``).
    :type sep: str

    :raises ValueError:
        If ``n_params`` is not 6, 12, or 24, or if required columns are missing.

    :return:
        DataFrame with motion regressors as columns.
    :rtype: pandas.DataFrame
    """

    if n_params not in [6, 12, 24]:
        raise ValueError("n_params must be 6, 12, or 24")
    conf_filename = glob.glob(os.path.join(rp_dir, f"rp_sSST_*.txt"))[0]
    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    motion_df = pd.read_csv(conf_filename, header=None, sep='\s+', names=motion_cols)

    if n_params == 6:
        return motion_df
    # Compute derivatives
    derivatives = motion_df.diff().fillna(0)
    derivatives.columns = [f"{col}_derivative" for col in motion_cols]
    motion_12_df = pd.concat([motion_df, derivatives], axis=1)

    if n_params == 12:
        return motion_12_df
        # For 24 parameters, square the original + derivative terms
    motion_12_squared = motion_12_df ** 2
    motion_12_squared.columns = [f"{col}_sq" for col in motion_12_df.columns]

    motion_24_df = pd.concat([motion_12_df, motion_12_squared], axis=1)
    return motion_24_df


def create_csf_confound(fmri_imgs, csf_mask, save_confounds=False):
    pass

def create_design_matrix(
    events_df: Optional[pd.DataFrame] = None,
    confounds_df: Optional[pd.DataFrame] = None,
    include_events: Optional[List[str]] = None,
    n_vols: int = 238,
    t_r: float = 2.0,
    hrf_model: str = 'spm + derivative + dispersion',
    drift_model: str = 'cosine',
    high_pass: float = 0.008
    ) -> pd.DataFrame:
    """
    Create a first-level design matrix for fMRI GLM analysis.

    :param events_df:
        DataFrame containing events information with at least columns:
        ``onset``, ``duration``, and ``trial_type``.
    :type events_df: pandas.DataFrame
    :param confounds_df:
        Optional DataFrame of confound regressors (e.g. motion parameters).
        Each column will be added as a regressor to the design matrix.
    :type confounds_df: Optional[pandas.DataFrame]
    :param include_events:
        Optional list of trial types to include. If provided, the events
        DataFrame will be filtered to only these trial types.
    :type include_events: Optional[List[str]]
    :param n_vols:
        Number of fMRI volumes (time points).
    :type n_vols: int
    :param t_r:
        Repetition time of the fMRI sequence in seconds.
    :type t_r: float
    :param hrf_model:
        Hemodynamic response function model name to use in the design matrix.
        E.g. ``'spm + derivative + dispersion'``.
    :type hrf_model: str
    :param drift_model:
        Low-frequency drift model. E.g. ``'cosine'``.
    :type drift_model: str
    :param high_pass:
        High-pass filter cutoff in Hz.
    :type high_pass: float

    :return:
        Design matrix as a pandas DataFrame with regressors as columns
        and time points as rows.
    :rtype: pandas.DataFrame
    """

    if confounds_df is not None:
        assert confounds_df.shape[0] == n_vols

    frame_times = np.linspace(t_r / 2, n_vols * t_r + t_r / 2, n_vols, endpoint=False)

    if (events_df is not None) and (include_events is not None):
        events_df = events_df[events_df['trial_type'].isin(include_events)].copy()

    dm = make_first_level_design_matrix(frame_times,
                                        events_df,
                                        drift_model=drift_model,
                                        high_pass=high_pass,
                                        hrf_model=hrf_model,
                                        add_regs=confounds_df
                                        )

    return dm

def fit_first_level_glm_with_dm(run_imgs,
                                design_matrices,
                                mask_img = None,
                                noise_model = 'ar',
                                minimize_memory = False,
                                return_residuals = True):
    t_r = 2
    slice_time_ref = 0.5
    hrf_model = 'spm + derivative + dispersion'
    drift_model = 'cosine'
    high_pass = 0.008  # cutoff: 0.008 Hz (i.e., 128 seconds)
    smoothing_fwhm = None

    flm = FirstLevelModel(
        t_r=t_r,
        slice_time_ref=slice_time_ref,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        mask_img=mask_img,
        smoothing_fwhm=smoothing_fwhm,
        noise_model=noise_model,
        minimize_memory=minimize_memory,
        verbose=True  # this will print out some useful info later
    )

    flm.fit(run_imgs=run_imgs, design_matrices=design_matrices)
    if return_residuals:
        return flm.residuals
    else:
        return flm
