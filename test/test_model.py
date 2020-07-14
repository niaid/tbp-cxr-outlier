from tbpcxr.model import PCAModel
import SimpleITK as sitk


def test_load_outlier_pcamodel():
    model = PCAModel.load_outlier_pcamodel()

    assert isinstance(model.image_atlas, sitk.Image)
    assert isinstance(model.image_ref, sitk.Image)
    assert all([s == PCAModel.SAMPLE_IMAGE_SIZE for s in model.image_ref.GetSize()])
    assert model.pca is not None
    assert model.outlier_detector is not None
