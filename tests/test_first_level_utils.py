import nibabel
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tmfc_py.first_level_utils import (create_motion_confounds_from_rp,
                                       create_design_matrix,
                                       fit_first_level_glm_with_dm)

from tmfc_py.matlab_utils import convert_3d_to_4d_fmri


PROJECT_ROOT = Path(__file__).parent.parent
REAL_DATA_PATH = PROJECT_ROOT / "data" / "raw"/"01_Sub"

@pytest.fixture
def fake_motion_folder(tmp_path):
    """
    Creates a temp file with 6 motion regressors,
    space-separated and NO header row.
    """
    # Create fake motion data
    n_timepoints = 10
    data = np.column_stack([
        np.linspace(0, 1, n_timepoints),  # trans_x
        np.linspace(1, 2, n_timepoints),  # trans_y
        np.linspace(-1, 1, n_timepoints),  # trans_z
        np.linspace(0.1, 0.5, n_timepoints),  # rot_x
        np.linspace(-0.5, 0.5, n_timepoints),  # rot_y
        np.linspace(0.0, 0.2, n_timepoints),  # rot_z
    ])

    # Write to a temporary file
    filename = f"rp_sSST_0001.txt"
    file_path = tmp_path / filename

    # Save without header, space-separated
    np.savetxt(file_path, data, fmt="%.6f", delimiter=' ')

    # Return the temp folder path
    return str(tmp_path)

def test_create_motion_confounds_from_rp(fake_motion_folder):

   conf_df  =  create_motion_confounds_from_rp(fake_motion_folder, n_params=6)
   assert conf_df.shape[1] == 6
   assert "rot_z" in conf_df.columns
   conf_df_deriv = create_motion_confounds_from_rp(fake_motion_folder, n_params=12)
   assert "rot_z_derivative" in conf_df_deriv.columns
   assert conf_df_deriv.shape[1] == 12
   conf_df_sq = create_motion_confounds_from_rp(fake_motion_folder, n_params=24)
   assert conf_df_sq.shape[1] == 24


@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data file not found")
def test_create_motion_confounds_real_data() -> None:

    conf_folder = str(REAL_DATA_PATH/"swar"/"02_Run")
    conf_df = create_motion_confounds_from_rp(conf_folder, n_params=6)
    assert conf_df.shape[1] == 6
    assert "rot_z" in conf_df.columns


@pytest.fixture
def fake_events_df():
    """
    Create a fake events dataframe for testing.
    """
    n_trials = 5
    events = pd.DataFrame({
        'onset': np.arange(0, n_trials * 20, 20),
        'duration': np.repeat(2.0, n_trials),
        'trial_type': ['taskA', 'taskB', 'taskA', 'taskC', 'taskB']
    })
    return events

def test_create_design_matrix_filter_events(fake_events_df, fake_motion_folder):
    conf_df_sq = create_motion_confounds_from_rp(fake_motion_folder, n_params=24)

    dm = create_design_matrix(fake_events_df,
                              conf_df_sq,
                              include_events=['taskA'],
                              n_vols=10)
    assert not any('taskB' in col for col in dm.columns)
    assert  any('taskA' in col for col in dm.columns)
    assert dm.shape[0] == 10




@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data file not found")
def test_create_design_matrix_real_data() -> None:
    conf_folder = str(REAL_DATA_PATH / "swar" / "02_Run")
    events_filename = str(REAL_DATA_PATH/"sots"/"02_Run_06_Deriv_[fix_onset]_[700ms_dur].csv")
    events_df = pd.read_csv(events_filename, sep='\t')
    conf_df = create_motion_confounds_from_rp(conf_folder, n_params=24)
    dm = create_design_matrix(events_df,
                              conf_df,
                              include_events=['Nuisance'],
                              n_vols=conf_df.shape[0])

    assert dm.shape[0] == conf_df.shape[0]


@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data file not found")
def test_fit_first_level_glm() -> None:

    conf_folder = str(REAL_DATA_PATH / "swar" / "02_Run")
    events_filename = str(REAL_DATA_PATH / "sots" / "02_Run_06_Deriv_[fix_onset]_[700ms_dur].csv")
    events_df = pd.read_csv(events_filename, sep='\t')
    conf_df = create_motion_confounds_from_rp(conf_folder, n_params=24)
    dm = create_design_matrix(events_df,
                              conf_df,
                              include_events=['Nuisance'],
                              n_vols=conf_df.shape[0])
    run_imgs = convert_3d_to_4d_fmri(conf_folder)
    flm = fit_first_level_glm_with_dm(run_imgs, dm, return_residuals=False, noise_model = 'ols')
    beta_map = flm.compute_contrast(
        'Nuisance',
        output_type="effect_size",
    )
    resids = fit_first_level_glm_with_dm(run_imgs, dm, return_residuals=True)

    nibabel.save(resids[0], "../data/interim/cleaned_vois_2run.nii.gz")

    assert True




