import SimpleITK as sitk
import numpy as np


def read_dcm(fname):
    """
    Read an x-ray DICOM file with the GDCMImageIO, reducing it to 2D from 3D as needed.

    :param fname: A DICOM filename
    :return: a 2D SimpleITK Image
    """
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetOutputPixelType(sitk.sitkFloat32)
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(fname)

    image_file_reader.ReadImageInformation()

    image_size = list(image_file_reader.GetSize())
    if len(image_size) == 3 and image_size[2] == 1:
        image_size[2] = 0
        image_file_reader.SetExtractSize(image_size)

    return image_file_reader.Execute()


def normalize_img(img, sample_size=64):

    dim = img.GetDimension()
    original_origin = img.GetOrigin()
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    img.SetDirection(np.identity(dim).ravel().tolist())

    max_physical_size = max([sz*sp for sz, sp in zip(original_size, original_spacing)])

    # move origin so that the corner ( continuous index -.5 ) is at the origin
    img.SetOrigin([sp*0.5 for sz, sp in zip(original_size, original_spacing)])

    tx = sitk.ScaleTransform(dim)
    tx.SetCenter((0,) * dim)
    scale = max_physical_size
    tx.SetScale((scale,) * dim)

    output_spacing = (1.0 / sample_size,) * dim
    output_size = [int(sample_size*sz*sp/max_physical_size) for sz, sp in zip(original_size, original_spacing)]
    output_origin = [0.5 / sample_size] * dim

    img = sitk.SmoothingRecursiveGaussian(img, [0.75*max_physical_size/sample_size]*dim)
    img = sitk.Normalize(img)
    img = sitk.Resample(img,
                        transform=tx,
                        size=output_size,
                        outputSpacing=output_spacing,
                        outputOrigin=output_origin,
                        outputDirection=np.identity(dim).ravel().tolist(),
                        useNearestNeighborExtrapolator=False
                        )

    # center the image at the zero-origin
    center = img.TransformContinuousIndexToPhysicalPoint([idx / 2.0 for idx in img.GetSize()])
    img.SetOrigin([o - c for o, c in zip(output_origin, center)])

    return img
