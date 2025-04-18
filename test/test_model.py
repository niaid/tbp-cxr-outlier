from tbpcxr.model import Model
import tbpcxr
import SimpleITK as sitk
import pytest
import os.path

path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.dirname(path_to_current_file)


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


model_fname = ["cxr-test-06c.pkl"]


@pytest.mark.parametrize(
    "model_name", tbpcxr.model_list + [os.path.join(current_directory, "data", fn) for fn in model_fname]
)
def test_load_model(model_name):
    model = Model.load_model(model_name)

    assert isinstance(model.image_atlas, sitk.Image)
    assert isinstance(model.image_reference, sitk.Image)
    assert all([s == model.reference_size for s in model.image_reference.GetSize()])
    assert model.pca is not None
    assert model.outlier_detector is not None

    img = model.register_to_atlas_and_resample(sitk.Noise(model.image_atlas))
    assert all([s == model.reference_size for s in img.GetSize()])


data_fname = [
    "00000041_005.png",
    "00000047_002.png",
    "00000050_001.png",
    "00000103_000.png",
    "00000109_003.png",
]


@pytest.mark.parametrize("filename, outlier_class", [pytest.param(fn, 1) for fn in data_fname])
def test_A(filename, outlier_class, outlier_pcamodel):
    path_to_file = os.path.join(current_directory, "data", filename)

    img = sitk.ReadImage(path_to_file, sitk.sitkFloat32)

    arr = outlier_pcamodel.to_observations([img])

    assert outlier_pcamodel.outlier_predictor(arr)[0] == outlier_class


def test_to_observations_constant(outlier_pcamodel):
    """
    Test the to_observations method with a constant image.
    """
    img = sitk.Image(224, 224, sitk.sitkFloat32)

    img += 1.2

    arr = outlier_pcamodel.to_observations([img])

    expected_size = (outlier_pcamodel.reference_size - 2 * outlier_pcamodel.reference_crop) ** 2

    assert arr.shape == (1, expected_size)
