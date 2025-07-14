from typing import Any, Generator
import pytest
import scipy.io as sio
import tempfile
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
from pathlib import Path
from tmfc_py.matlab_utils import create_events_from_mat, convert_3d_to_4d_fmri

PROJECT_ROOT = Path(__file__).parent.parent
REAL_DATA_PATH = PROJECT_ROOT / "data" / "raw"/"01_Sub/sots/02_Run_06_Deriv_[fix_onset]_[700ms_dur].mat"

@pytest.fixture
def temp_mat_file() -> Generator[str, Any, None]:
    """Temporary matlab file."""
    data = {
        'onsets': np.array([[0., 1., 2.], [4., 6.]], dtype=object),
        'durations': np.array([[1., 1., 1.], [2., 2.]], dtype=object),
        'names': np.array(['event1', 'event2'], dtype=object)
    }
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        sio.savemat(tmp.name, data)
        yield tmp.name
    Path(tmp.name).unlink()


def test_create_events_from_mat_correct_output(temp_mat_file: str) -> None:
    """Test correct processing of a .mat file with multiple events."""
    result = create_events_from_mat(temp_mat_file)
    expected = pd.DataFrame({
        'trial_type': ['event1', 'event1', 'event1', 'event2', 'event2'],
        'onset': [0., 1., 2., 4., 6.],
        'duration': [1., 1., 1., 2., 2.]
    }).sort_values(by='onset')
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="Real data file not found")
def test_create_events_from_mat_real_data() -> None:
    """Test that create_events_from_mat works correctly.
    Pass only when real data is available.
    """
    result = create_events_from_mat(str(REAL_DATA_PATH))

    assert isinstance(result, pd.DataFrame), "Result must be a pandas DataFrame"
    assert set(result.columns) == {"trial_type", "onset", "duration"}, "DataFrame must have correct columns"
    assert (result["onset"] >= 0).all(), "All onsets must be non-negative"
    assert result["onset"].is_monotonic_increasing, "DataFrame must be sorted by onset"
    assert not result.empty, "DataFrame should not be empty for real data"
    assert result["trial_type"].notnull().all(), "Trial types must not contain null values"
    assert result["duration"].notnull().all(), "Durations must not contain null values"


@pytest.fixture
def nifti_file_factory(tmp_path):
    """
    A pytest fixture that returns a factory function.
    This factory can be used within tests to create temporary 3D NIfTI files.
    """

    def _create_files(filenames, shape=(10, 10, 10), affine=np.eye(4)):
        """Creates NIfTI files with specified names and properties."""
        created_files = []
        for filename in filenames:
            # Create dummy data
            data = np.random.randn(*shape)
            img = nib.Nifti1Image(data, affine)

            # Save to the temporary directory provided by pytest's tmp_path
            filepath = tmp_path / filename
            nib.save(img, filepath)
            created_files.append(str(filepath))
        return created_files

    return _create_files


def test_successful_concatenation_and_sorting(nifti_file_factory, tmp_path):
    """
    Tests the happy path: correctly named files are found, sorted, and concatenated.
    """
    # Create files out of order to test the sorting logic
    filenames = [
        "swarsSST_000003-01.nii",
        "swarsSST_000001-01.nii",
        "swarsSST_000002-01.nii",
    ]
    nifti_file_factory(filenames)

    # Run the function under test
    result_4d_img = convert_3d_to_4d_fmri(str(tmp_path))

    # Assertions
    assert isinstance(result_4d_img, nib.Nifti1Image)
    assert result_4d_img.shape == (10, 10, 10, 3)  # (x, y, z, time)

    # Verify the sorting was correct by checking the data
    data_vol1 = image.get_data(str(tmp_path / "swarsSST_000001-01.nii"))
    assert np.allclose(image.get_data(result_4d_img)[..., 0], data_vol1)

def test_no_files_found_raises_error(tmp_path):
    """
    Tests that FileNotFoundError is raised when no matching files are in the directory.
    """
    with pytest.raises(FileNotFoundError, match="No NIfTI files found"):
        convert_3d_to_4d_fmri(str(tmp_path))

REAL_NII_PATH = PROJECT_ROOT / "data" / "raw"/"01_Sub/swar/02_Run"

@pytest.mark.skipif(not REAL_NII_PATH.exists(), reason=f"Real data directory not found at: {REAL_NII_PATH}")
def test_with_real_data():
    """
    An integration test using real data. This test is skipped if the data is not present.
    It checks if the function can run without errors on a real-world file structure.
    """
    print(f"\nRunning integration test with real data from: {REAL_NII_PATH}")

    # Run the function on the real data directory
    # Assuming the default file pattern "swarsSST_*.nii" matches your real data
    result_4d_img = convert_3d_to_4d_fmri(REAL_NII_PATH)

    # Basic assertions for an integration test
    assert isinstance(result_4d_img, nib.Nifti1Image)
    assert result_4d_img.ndim == 4
    assert result_4d_img.shape[3] > 1  # Check that it actually concatenated multiple volumes
    print(f"Successfully created 4D image with shape: {result_4d_img.shape}")