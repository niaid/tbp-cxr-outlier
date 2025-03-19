import SimpleITK as sitk
import numpy as np

try:
    from rap_sitkcore import read_dcm
except ImportError:

    def read_dcm(filename: str) -> sitk.Image:
        """
        Read an x-ray DICOM file with GDCMImageIO, reducing it to 2D from 3D as needed.
        Only the SimpleITK GDCM reader is tried.

        :param filename: A DICOM filename
        :return: a 2D SimpleITK Image
        """

        image_file_reader = sitk.ImageFileReader()
        image_file_reader.SetOutputPixelType(sitk.sitkFloat32)
        image_file_reader.SetImageIO("GDCMImageIO")
        image_file_reader.SetFileName(filename)

        image_file_reader.ReadImageInformation()

        image_size = list(image_file_reader.GetSize())
        if len(image_size) == 3 and image_size[2] == 1:
            image_size[2] = 0
            image_file_reader.SetExtractSize(image_size)

        return image_file_reader.Execute()


def _auto_crop(img: sitk.Image) -> sitk.Image:
    """
    Automatically crop the image to the bounding box of the foreground defined by OtsuThreshold.

    :param img: The input image
    :return: The cropped image
    """

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask = otsu_filter.Execute(img)

    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(mask)
    boundingBox = stats_filter.GetBoundingBox(1)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetRegionOfInterest(boundingBox)

    return roi.Execute(img)


def normalize_img(
    image: sitk.Image, sample_size: int = 64, smoothing_sigma_in_output_pixels: float = 0.75, auto_crop: bool = True
) -> sitk.Image:
    """
    The input image is resampled with translation and scaling to fit into a unit square centered at the origin. The
    physical extend of the output is within [-0.5, 0.5]. The aspect ratio of the input image is preserved. The input's
    direction cosine matrix is ignored. The image intensities are normalized to a have a mean of 0, and a standard
    deviation of 1.

    :param image: The input image
    :param sample_size: The maximum number of pixels in an axis. It will be less if input's physical size is not 1:1.
    :param smoothing_sigma_in_output_pixels: Before resample Gaussian smoothing is performed, with a sigma equivalent to
    :param auto_crop: If True, the image is cropped to the bounding box of the foreground defined by OtsuThreshold.
    :return: An image
    """

    if auto_crop:
        image = _auto_crop(image)

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
