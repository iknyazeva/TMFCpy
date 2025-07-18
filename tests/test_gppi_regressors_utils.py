import pytest
from pathlib import Path
import numpy as np
from nibabel import Nifti1Image
from tmfc_py.gppi_regressors_utils import (
    extract_vois_from_clean_img,
    _load_nifti,
    _validate_inputs,
    _extract_single_mask_time_series
)
import scipy.io as sio


PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_IMG_PATH = PROJECT_ROOT/'data'/'interim'/'cleaned_vois_2run6p_ar.nii.gz'
VOI_MASK_PATH = PROJECT_ROOT/'data'/'external'/'01_Sub'/'MaskedRois'/'001_L_V1_masked.nii'
VOI_SPM = PROJECT_ROOT/'data'/'external'/'01_Sub'/'VOIs'/'VOI_001_L_V1_1.mat'


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
    spm_time_series = sio.loadmat(str(VOI_SPM))['Y'].squeeze()
    assert time_series.shape[0] == clean_img.shape[-1]

def test_extract_vois_from_clean_img():
    assert False
