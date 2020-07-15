from tbpcxr.model import PCAModel
from tbpcxr.utilities import normalize_img
import SimpleITK as sitk
import pytest
import os.path


@pytest.fixture
def outlier_pcamodel():
    return PCAModel.load_outlier_pcamodel()


def test_load_outlier_pcamodel():
    model = PCAModel.load_outlier_pcamodel()

    assert isinstance(model.image_atlas, sitk.Image)
    assert isinstance(model.image_ref, sitk.Image)
    assert all([s == PCAModel.SAMPLE_IMAGE_SIZE for s in model.image_ref.GetSize()])
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

    nimg = normalize_img(img)

    rimg = outlier_pcamodel.register_to_atlas_and_resample(nimg, verbose=0)
    arr = outlier_pcamodel._images_to_arr([rimg])

    assert outlier_pcamodel.outlier_predictor(arr)[0] == outlier_class
