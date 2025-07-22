import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from tmfc_py.gppi_regressors_utils import (
    create_ppi_regressor,
    create_raw_psych_regressor,
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



class TestCreatePPIRegressor:
    """Test suite for the create_ppi_regressor function."""

    @pytest.fixture
    def events_df(self):
        """Provide a sample events DataFrame."""
        return pd.DataFrame({
            'onset': [2.0, 10.0],
            'duration': [4.0, 4.0],
            'trial_type': ['taskA', 'taskA']
        })

    @pytest.fixture
    def voi_bold(self):
        """Provide a sample BOLD signal."""
        return np.random.randn(100)

    def test_valid_input(self, voi_bold, events_df):
        """Test create_ppi_regressor with valid inputs."""
        ppi_regressor = create_ppi_regressor(
            voi_bold=voi_bold,
            events_df=events_df,
            task_name='taskA',
            tr=2.0,
            microtime_resolution=16
        )
        assert ppi_regressor.shape == (100,), "Output shape should match voi_bold length"
        assert np.all(np.isfinite(ppi_regressor)), "Output should contain finite values"

    def test_empty_task(self, voi_bold, events_df):
        """Test create_ppi_regressor with non-existent task name."""
        ppi_regressor = create_ppi_regressor(
            voi_bold=voi_bold,
            events_df=events_df,
            task_name='taskB',
            tr=2.0,
            microtime_resolution=16
        )
        assert ppi_regressor.shape == (100,), "Output shape should match voi_bold length"
        assert np.all(ppi_regressor == 0), "Output should be zero for empty task"
    def test_empty_voi_bold(self, events_df):
        """Test create_ppi_regressor with empty voi_bold."""
        with pytest.raises(ValueError, match="voi_bold is empty."):
            create_ppi_regressor(
                voi_bold=np.array([]),
                events_df=events_df,
                task_name='taskA',
                tr=2.0,
                microtime_resolution=16
            )
    def test_short_duration_events(self, voi_bold):
        """Test create_ppi_regressor with short-duration events."""
        short_events_df = pd.DataFrame({
            'onset': [2.0],
            'duration': [0.1],
            'trial_type': ['taskA']
        })
        ppi_regressor = create_ppi_regressor(
            voi_bold=voi_bold,
            events_df=short_events_df,
            task_name='taskA',
            tr=2.0,
            microtime_resolution=16
        )
        assert ppi_regressor.shape == (100,), "Output shape should match voi_bold length"
        assert np.all(np.isfinite(ppi_regressor)), "Output should contain finite values"

class TestExtractVoisFromCleanImg:
    @pytest.fixture
    def clean_img(self):
        return str(CLEAN_IMG_PATH)


    @pytest.fixture
    def voi_mask(self):
        return str(VOI_MASK_PATH)


    def test_load_nifti(self, clean_img):
        img = _load_nifti(clean_img)
        assert isinstance(img, Nifti1Image)


    @pytest.mark.skipif(not CLEAN_IMG_PATH.exists(), reason="Real data file not found")
    def test_load_nifti(self, clean_img):
        img = _load_nifti(clean_img)
        assert isinstance(img, Nifti1Image)


    @pytest.mark.skipif(not (CLEAN_IMG_PATH.exists()
                             or VOI_MASK_PATH.exists()), reason="Real data file not found")
    def test_validate_inputs(self, clean_img, voi_mask):
        clean_img = _load_nifti(clean_img)
        _validate_inputs(clean_img, seed_mask=voi_mask, seed_coords=None, agg='eig')
        with pytest.raises(ValueError):
            _validate_inputs(clean_img, None, None, 'eig')


    def test_extract_single_mask_time_series(self,clean_img, voi_mask):
        clean_img = _load_nifti(clean_img)
        mask = _load_nifti(voi_mask)
        time_series = _extract_single_mask_time_series(clean_img, mask, 'eig')
        time_series_spm = sio.loadmat(str(VOI_SPM))['Y'].squeeze()
        r = np.corrcoef(time_series, time_series_spm)[0, 1]
        assert r > 0.2
        assert time_series.shape[0] == clean_img.shape[-1]


    def test_extract_vois_from_clean_img(self, clean_img, voi_mask):
        time_series = extract_vois_from_clean_img(clean_img, seed_mask=voi_mask)
        assert time_series.shape[0] == clean_img.shape[-1]



class TestCreateRawPsychRegressor:
    """Test suite for the create_raw_psych_regressor function."""

    @pytest.fixture
    def events_df(self):

        events_df = pd.DataFrame({
            'onset': [10, 50, 90, 130, 20, 60, 100, 140],
            'duration': [15, 15, 15, 15, 8, 8, 8, 8],
            'trial_type': ['TaskA', 'TaskA', 'TaskA', 'TaskA', 'TaskB', 'TaskB', 'TaskB', 'TaskB']
        })
        return events_df

    @pytest.fixture
    def events_df_with_amplitude(self):
        """Provide a sample events DataFrame with amplitude column."""
        return pd.DataFrame({
            'onset': [2.0, 10.0],
            'duration': [4.0, 4.0],
            'trial_type': ['taskA', 'taskA'],
            'amplitude': [1.5, 2.0]
        })


    def test_create_raw_psych_regressor(self, events_df):
        n_vols, tr, microtime_resolution = 180, 2, 32
        psy_reg = create_raw_psych_regressor(events_df, 'TaskA', n_vols, tr, microtime_resolution)
        assert psy_reg.shape[0] == n_vols*microtime_resolution
        assert psy_reg[0] == 0
        assert psy_reg[5*32] == 1

    def test_parametric_modulation(self, events_df_with_amplitude):
        """Test create_raw_psych_regressor with amplitude column."""
        regressor = create_raw_psych_regressor(
            events_df=events_df_with_amplitude,
            task_name='taskA',
            n_vols=100,
            tr=2.0,
            microtime_resolution=16,
            amplitude_col='amplitude'
        )
        assert regressor.shape == (100 * 16,), "Output shape should be n_vols * microtime_resolution"
        microtime_tr = 2.0 / 16
        start_idx1 = int(2.0 / microtime_tr)
        end_idx1 = int(6.0 / microtime_tr)
        start_idx2 = int(10.0 / microtime_tr)
        end_idx2 = int(14.0 / microtime_tr)
        assert np.allclose(regressor[start_idx1:end_idx1], 1.5), "First event should have amplitude 1.5"
        assert np.allclose(regressor[start_idx2:end_idx2], 2.0), "Second event should have amplitude 2.0"
        assert np.sum(regressor != 0) > 0, "Regressor should have non-zero values"

    def test_empty_task(self, events_df):
        """Test create_raw_psych_regressor with non-existent task name."""
        regressor = create_raw_psych_regressor(
            events_df=events_df,
            task_name='taskB',
            n_vols=100,
            tr=2.0,
            microtime_resolution=16
        )
        assert regressor.shape == (100 * 16,), "Output shape should be n_vols * microtime_resolution"
        assert np.all(regressor == 0), "Output should be zero for empty task"


    @pytest.mark.skipif(not EVENTS_PATH.exists(), reason="Real data file not found")
    def test_create_raw_psych_regressor_real(self):
        events_df = pd.read_csv(str(EVENTS_PATH), sep='\t')
        psy_reg = create_raw_psych_regressor(events_df, 'GO', 231, 2, 32)
        assert psy_reg.shape[0] == 231*32
        assert psy_reg[13] == 1




