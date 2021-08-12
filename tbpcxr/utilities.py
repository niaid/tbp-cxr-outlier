import SimpleITK as sitk
import numpy as np
from rap_sitkcore import read_dcm


def normalize_img(
    image: sitk.Image, sample_size: int = 64, smoothing_sigma_in_output_pixels: float = 0.75
) -> sitk.Image:
    """
    The input image is resampled with translation and scaling to fit into a unit square centered at the origin. The
    physical extend of the output is within [-0.5, 0.5]. The aspect ratio of the input image is preserved. The input's
    direction cosine matrix is ignored. The image intensities are normalized to a have a mean of 0, and a standard
    deviation of 1.

    :param image: The input image
    :param sample_size: The maximum number of pixels in an axis. It will be less if input's physical size is not 1:1.
    :param smoothing_sigma_in_output_pixels: Before resample Gaussian smoothing is performed, with a sigma equivalent to
    :return: An im
    """

    dim = image.GetDimension()
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    image.SetDirection(np.identity(dim).ravel().tolist())

    max_physical_size = max([sz * sp for sz, sp in zip(original_size, original_spacing)])

    # move origin so that the corner ( continuous index -.5 ) is at the origin
    image.SetOrigin([sp * 0.5 for sz, sp in zip(original_size, original_spacing)])

    tx = sitk.ScaleTransform(dim)
    tx.SetCenter((0,) * dim)
    scale = max_physical_size
    tx.SetScale((scale,) * dim)

    output_spacing = (1.0 / sample_size,) * dim
    output_size = [int(sample_size * sz * sp / max_physical_size) for sz, sp in zip(original_size, original_spacing)]
    output_origin = [0.5 / sample_size] * dim

    image = sitk.SmoothingRecursiveGaussian(
        image, [smoothing_sigma_in_output_pixels * max_physical_size / sample_size] * dim
    )
    image = sitk.Normalize(image)
    image = sitk.Resample(
        image,
        transform=tx,
        size=output_size,
        outputSpacing=output_spacing,
        outputOrigin=output_origin,
        outputDirection=np.identity(dim).ravel().tolist(),
        useNearestNeighborExtrapolator=False,
    )

    # center the image at the zero-origin
    center = image.TransformContinuousIndexToPhysicalPoint([idx / 2.0 for idx in image.GetSize()])
    image.SetOrigin([o - c for o, c in zip(output_origin, center)])

    return image
