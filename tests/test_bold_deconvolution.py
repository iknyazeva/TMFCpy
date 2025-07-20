import timeit
import pytest
import numpy as np
from tmfc_py.bold_deconvolution import ridge_regress_deconvolution
from tmfc_py.bold_deconvolution import  dctmtx_numpy_vect, compute_xb_Hxb
import scipy.io as sio
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path


# Assuming the ridge_regress_deconvolution function is already imported
PROJECT_ROOT = Path(__file__).parent.parent
VOI_SPM = PROJECT_ROOT / 'data' / 'external' / '01_Sub' / 'VOIs' / 'VOI_001_L_V1_1.mat'


@pytest.fixture(scope="class")
def test_data():
    """
    Provides common setup for all tests in the class.
    scope="class" means this fixture will be set up once per class.
    """
    data =  sio.loadmat(str(VOI_SPM))['Y'].squeeze()

    return {
        "BOLD": data,
        "TR": 2,
        "alpha": 0.005,
        "NT": 32
    }
@pytest.mark.usefixtures("test_data")
class TestRidgeRegressDeconvolution:

    def test_basic_functionality(self, test_data):
        # Test if the function returns an output of expected shape and type
        neuro = ridge_regress_deconvolution(test_data["BOLD"],
                                            test_data["TR"],
                                            test_data["alpha"],
                                            test_data["NT"])
        assert isinstance(neuro, np.ndarray)


    def test_empty_input(self):
        # Test if the function raises ValueError for empty BOLD signal
        with pytest.raises(ValueError):
            ridge_regress_deconvolution(np.array([]), 2, 0.05, 32)

    def test_zero_TR(self, test_data):
        # Test if the function raises ZeroDivisionError for TR = 0
        with pytest.raises(ZeroDivisionError):
            ridge_regress_deconvolution(test_data["BOLD"],
                                        0,
                                        test_data["alpha"],
                                        test_data["NT"])


