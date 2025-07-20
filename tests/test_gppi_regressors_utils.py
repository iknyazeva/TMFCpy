import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from tmfc_py.gppi_regressors_utils import (
    extract_vois_from_clean_img,
    create_raw_psych_regressor,
    _load_nifti,
    _validate_inputs,
    _extract_single_mask_time_series
)
import scipy.io as sio

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_IMG_PATH = PROJECT_ROOT / 'data' / 'interim' / 'cleaned_vois_2run6p_ar.nii.gz'
VOI_MASK_PATH = PROJECT_ROOT / 'data' / 'external' / '01_Sub' / 'MaskedRois' / '001_L_V1_masked.nii'
VOI_SPM = PROJECT_ROOT / 'data' / 'external' / '01_Sub' / 'VOIs' / 'VOI_001_L_V1_1.mat'
EVENTS_PATH = PROJECT_ROOT /'data'/'raw'/ '01_Sub' / 'sots'/'02_Run_06_Deriv_[fix_onset]_[700ms_dur].csv'


@pytest.fixture
def clean_img():
    return str(CLEAN_IMG_PATH)


@pytest.fixture
def voi_mask():
    return str(VOI_MASK_PATH)


def test_load_nifti(clean_img):
    img = _load_nifti(clean_img)
    assert isinstance(img, Nifti1Image)


@pytest.mark.skipif(not CLEAN_IMG_PATH.exists(), reason="Real data file not found")
def test_load_nifti(clean_img):
    img = _load_nifti(clean_img)
    assert isinstance(img, Nifti1Image)


@pytest.mark.skipif(not (CLEAN_IMG_PATH.exists()
                         or VOI_MASK_PATH.exists()), reason="Real data file not found")
def test_validate_inputs(clean_img, voi_mask):
    clean_img = _load_nifti(clean_img)
    _validate_inputs(clean_img, seed_mask=voi_mask, seed_coords=None, agg='eig')
    with pytest.raises(ValueError):
        _validate_inputs(clean_img, None, None, 'eig')


def test_extract_single_mask_time_series(clean_img, voi_mask):
    clean_img = _load_nifti(clean_img)
    mask = _load_nifti(voi_mask)
    time_series = _extract_single_mask_time_series(clean_img, mask, 'eig')
    time_series_spm = sio.loadmat(str(VOI_SPM))['Y'].squeeze()
    r = np.corrcoef(time_series, time_series_spm)[0, 1]
    assert r > 0.2
    assert time_series.shape[0] == clean_img.shape[-1]


def test_extract_vois_from_clean_img():
    assert False


@pytest.fixture
def sample_events_df():
    n_vols = 180
    tr = 2.0
    microtime_resolution = 32

    events_df = pd.DataFrame({
        'onset': [10, 50, 90, 130, 20, 60, 100, 140],
        'duration': [15, 15, 15, 15, 8, 8, 8, 8],
        'trial_type': ['TaskA', 'TaskA', 'TaskA', 'TaskA', 'TaskB', 'TaskB', 'TaskB', 'TaskB']
    })
    return events_df, n_vols, tr, microtime_resolution


def test_create_raw_psych_regressor(sample_events_df):
    events_df, n_vols, tr, microtime_resolution = sample_events_df
    psy_reg = create_raw_psych_regressor(events_df, 'TaskA', n_vols, tr, microtime_resolution)
    assert psy_reg.shape[0] == n_vols*microtime_resolution
    assert psy_reg[0] == 0
    assert psy_reg[5*32] == 1


@pytest.mark.skipif(not EVENTS_PATH.exists(), reason="Real data file not found")
def test_create_raw_psych_regressor_real():
    events_df = pd.read_csv(str(EVENTS_PATH), sep='\t')
    psy_reg = create_raw_psych_regressor(events_df, 'GO', 231, 2, 32)
    assert psy_reg.shape[0] == 231*32
    assert psy_reg[13] == 1




