import SimpleITK as sitk
from typing import Iterable, Tuple


def cxr_affine(
    fixed: sitk.Image, moving: sitk.Image, verbose: int = 0, max_iterations: int = 500
) -> Tuple[sitk.Transform, float]:
    """
    Perform affine registration between two CXR images. First a 2D similarity transform is optimized
    then an affine transform.

    :param fixed: The fixed image, sample points are mapped from the fixed image to the moving
    :param moving: The moving image.
    :param verbose: 0 - no output, 1 - staged registration results, 2 - output each iteration
    :param max_iterations: The maximum number of optimization iteration to perform at each phase.
    :return: Returns tuple of the optimized affine transform mapping points from the fixed image to the moving image,
        and the final optimized metric value.
    """

    def command_iteration(method):
        print(
            "{0:3} = {1:7.5f} : {2} #{3}".format(
                method.GetOptimizerIteration(),
                method.GetMetricValue(),
                method.GetOptimizerPosition(),
                method.GetMetricNumberOfValidPoints(),
            )
        )

    tx = sitk.Similarity2DTransform()
    tx.SetCenter([0, 0])

    R = sitk.ImageRegistrationMethod()

    R.SetShrinkFactorsPerLevel([2])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetSmoothingSigmasPerLevel([1 / 32.0])

    R.SetMetricAsMattesMutualInformation()
    # R.SetMetricAsCorrelation()
    # R.SetMetricMovingMask(moving!=0)
    # R.SetOptimizerAsGradientDescent(learningRate=0.9,
    #                                numberOfIterations=500,
    #                                convergenceMinimumValue=1e-6 )

    R.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.9, lineSearchUpperLimit=1.5, numberOfIterations=max_iterations, convergenceMinimumValue=1e-5
    )
    R.SetOptimizerScalesFromIndexShift()
    R.SetInterpolator(sitk.sitkLinear)

    if verbose >= 2:
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    R.SetInitialTransform(tx, inPlace=True)

    outTx = R.Execute(fixed, moving)

    tx2 = sitk.AffineTransform(2)
    tx2.SetTranslation(tx.GetTranslation())
    tx2.SetCenter(tx.GetCenter())
    tx2.SetMatrix(tx.GetMatrix())

    R.SetShrinkFactorsPerLevel([2, 1])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetSmoothingSigmasPerLevel([1 / 64.0, 1 / 64.0])

    R.SetInitialTransform(tx2, inPlace=True)

    # tx = sitk.Similarity2DTransform()
    # tx.SetCenter([0.0,0.0])
    outTx = R.Execute(fixed, moving)

    if verbose >= 1:
        print(outTx)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

    return outTx, R.GetMetricValue()


def resample(fixed: sitk.Image, moving: sitk.Image, transform: sitk.Transform, verbose=0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.UseNearestNeighborExtrapolatorOn()

    return resampler.Execute(moving)


def avg_resample(fixed: sitk.Image, images: Iterable[sitk.Image]) -> sitk.Image:
    """
    Resamples all images onto the fixed image's a coordinate frame, ignoring the fixed image's values. The resampled
    images are accumulated and averaged for the results.

    :param fixed:
    :param images:
    :return:
    """

    avg = sitk.Cast(fixed, sitk.sitkFloat32) * 0.0
    for img in images:
        avg += sitk.Resample(img, avg, outputPixelType=sitk.sitkFloat32)

    avg /= len(images)
    return avg


def build_atlas(
    fixed: sitk.Image,
    images: Iterable[sitk.Image],
    fixed_crop_percent=0.10,
    verbose=0,
    max_iterations=500,
    register_repeat=2,
) -> sitk.Image:
    """
    Builds an CXR atlas ( average ) by repeatedly registering a list of images to an average, then updating the average.

    :param fixed:
    :param images:
    :param fixed_crop_percent:
    :param verbose:
    :param max_iterations: The maximum number of iterations per image stage
    :param register_repeat: The number of time all the images are registered to the average.
    :return:
    """

    avg = avg_resample(fixed, images)

    crop_size = [int(s * fixed_crop_percent) for s in fixed.GetSize()]

    for iter in range(register_repeat):
        if verbose >= 1:
            print("Atlas Iteration {}".format(iter))
        if verbose >= 2:
            filename = "build_atlas_{}.nrrd".format(iter)
            print("\tWriting {0}".format(filename))
            sitk.WriteImage(avg, filename)

        fixed_crop = sitk.Crop(avg, crop_size, crop_size)

        regs = []
        for moving_img in images:
            try:
                transform, metric_value = cxr_affine(
                    fixed_crop, moving=moving_img, verbose=verbose, max_iterations=max_iterations
                )
            except RuntimeError as e:
                print("Registration Error:")
                print(e)
                transform = sitk.TranslationTransform(2)

            regs.append(resample(avg, moving_img, transform, verbose=verbose - 1))

        avg = sitk.NaryAdd(regs)
        avg /= len(regs)

    return avg
