from tbpcxr.model import PCAModel, Model
from tbpcxr.utilities import normalize_img
import SimpleITK as sitk
import pytest
import os.path


@pytest.fixture
def outlier_pcamodel():
    return Model.load_outlier_pcamodel()


def test_load_outlier_pcamodel():
    model = Model.load_outlier_pcamodel()

    assert isinstance(model.image_atlas, sitk.Image)
    assert isinstance(model.image_reference, sitk.Image)
    assert all([s == model.reference_size for s in model.image_reference.GetSize()])
    assert model.pca is not None
    assert model.outlier_detector is not None


data_fname = [
    "00000041_005.png",
    "00000047_002.png",
    "00000050_001.png",
    "00000103_000.png",
    "00000109_003.png",
]


@pytest.mark.parametrize("filename, outlier_class",
                         [pytest.param(fn, 1) for fn in data_fname])
def test_A(filename, outlier_class, outlier_pcamodel):

    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(path_to_current_file)
    path_to_file = os.path.join(current_directory, "data", filename)

    img = sitk.ReadImage(path_to_file, sitk.sitkFloat32)

    arr = outlier_pcamodel.to_observations([img])


    assert outlier_pcamodel.outlier_predictor(arr)[0] == outlier_class
