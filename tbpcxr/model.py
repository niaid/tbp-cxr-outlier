import pkg_resources as res

from sklearn import decomposition
import SimpleITK as sitk
from typing import List

import numpy as np

from sklearn.covariance import MinCovDet
from sklearn.covariance import EllipticEnvelope

import pickle

from . import registration


class PCAModel:
    """
    This class represents a model of an expected image to detect if another image conforms to the model or is an
    outlier.

    Four states of the image data need to be defined:
     * normalized image: An sitk.Image which has been registered to the :attr:`image_atlas`, resampled onto the space
       of the :attr:`image_ref` then the intensities normalized.
     * observation vector : A normalized image converted into an array ( np.ndarray ).
     * feature vector: A representation of an image that has been converted into an observation then further reduced.

    """

    SAMPLE_IMAGE_SIZE = 64
    """The reference image is defined as this number of pixels in each dimension."""

    CROP_SIZE = 6
    """The reference image, image atlas, and input images are cropped by this amount on each edge for registration, and
    the PCA feature vector."""

    def __init__(self):
        self.pca = None
        """The PCA decomposition of the cropped input image as a vector."""

        self.pca_cov = None
        """The object used to estimate the covariance of the training data in the PCA feature space."""

        self.outlier_detector = None
        """The trained object used to classify a fitted image as a PCA feature vector."""

        self.image_atlas = None
        """The atlas image which input images are registered too.

        This image must have the same physical extent as image_ref

        :type: sitk.Image or None
        """

        dim = 2
        self.image_ref = sitk.Image([self.SAMPLE_IMAGE_SIZE]*dim, sitk.sitkUInt8)
        self.image_ref.SetOrigin([-0.5 + (0.5 / self.SAMPLE_IMAGE_SIZE)]*dim)
        self.image_ref.SetSpacing([1.0 / self.SAMPLE_IMAGE_SIZE]*dim)

    def compute(self, observation_arr: np.ndarray, components: int, contamination=0.10) -> None:
        """
        Computes the PCA decomposition, and outlier classifier from the input feature array.

        :param observation_arr: A set of images converted to an array of image observation vectors.
        :param components:
        :param contamination:
        :return:
        """
        # observation_arr is an array of ( n, feature_vector, )

        self.pca = decomposition.PCA(n_components=components)

        self.pca.fit(observation_arr)

        reduced_cxr = self.pca.transform(observation_arr)

        self.pca_cov = MinCovDet().fit(reduced_cxr)

        train_res = self.residuals(observation_arr)
        train_dist = self.robust_distance(observation_arr)

        X = np.stack((train_res, train_dist), axis=-1)

        self.outlier_detector = EllipticEnvelope(contamination=contamination)
        self.outlier_detector.fit(X)

    def register_to_atlas_and_resample(self, image: sitk.Image, verbose: int = 0) \
            -> sitk.Image:
        """
        Register to the input images to the atlas image, and resample to the reference image coordinates.

        :param image:
        :param verbose:
        :return: The registered and resampled image.
        """

        crop_size = [int(s * self.CROP_SIZE/self.SAMPLE_IMAGE_SIZE) for s in self.image_atlas.GetSize()]
        fixed_crop = sitk.Crop(self.image_atlas, crop_size, crop_size)

        try:
            transform, metric_value = registration.cxr_affine(fixed_crop, moving=image, verbose=verbose)
        except RuntimeError as e:
            print("Registration Error:")
            print(e)
            transform = sitk.TranslationTransform(2)

        return sitk.Resample(image,
                             referenceImage=self.image_ref,
                             transform=transform,
                             outputPixelType=sitk.sitkFloat32)

    def outlier_predictor(self, observation_arr: np.ndarray) -> List[int]:
        """
        Use the trained PCA and outlier classifier from the :meth:`compute` to determine if an array of observation
        vectors are outliers when compared to the input data set.

        :param observation_arr: An array of image observations
        :return: An array of results of outlier classification, 1 is an inlier and -1 is a classifier.
        """

        residuals = self.residuals(observation_arr)
        dist = self.robust_distance(observation_arr)

        X = np.stack((residuals, dist), axis=-1)
        return self.outlier_detector.predict(X)

    def residual_images(self, observation_arr: np.ndarray) -> List[sitk.Image]:
        """
        Compute the difference between the input observations and reduced PCA representation.

        :param observation_arr:
        :return:
        """
        reduced_cxr = self.pca.transform(observation_arr)
        restored_cxr = self.pca.inverse_transform(reduced_cxr)
        residual = (restored_cxr - observation_arr) ** 2
        return self._arr_to_images(residual)

    def restored_images(self, observation_arr: np.ndarray) -> List[sitk.Image]:
        """
        Compute the reduced PCA representation, then restore observation as an image representation.

        :param observation_arr:
        :return:
        """

        reduced_cxr = self.pca.transform(observation_arr)
        restored_cxr = self.pca.inverse_transform(reduced_cxr)
        return self._arr_to_images(restored_cxr)

    def residuals(self, observation_arr: np.ndarray) -> float:
        """
        Compute the root mean squared of the difference between the original observation and the restored observation.
        :param observation_arr:
        :return:
        """
        reduced_cxr = self.pca.transform(observation_arr)
        restored_cxr = self.pca.inverse_transform(reduced_cxr)

        # Compute root mean squared of residual image vectors
        return np.sqrt(np.mean((restored_cxr - observation_arr) ** 2, axis=1))

    def robust_distance(self, observation_arr: np.ndarray) -> float:
        """
        Compute the Mahalanobis distance in the PCA space for observations

        :param observation_arr:
        :return:
        """
        reduced_cxr = self.pca.transform(observation_arr)

        return np.sqrt(self.pca_cov.mahalanobis(reduced_cxr))

    def _arr_to_images(self, observation_arr: np.ndarray) -> List[sitk.Image]:
        """

        :param observation_arr:
        :return:
        """
        number_of_images = observation_arr.shape[0]
        size = self.SAMPLE_IMAGE_SIZE - 2 * self.CROP_SIZE
        return [sitk.GetImageFromArray(observation_arr[i, :].reshape(size, size))
                for i in range(number_of_images)]

    def _images_to_arr(self, imgs: List[sitk.Image]) -> np.ndarray:
        crop_size = [self.CROP_SIZE]*2
        return np.stack([sitk.GetArrayFromImage(sitk.Crop(img, crop_size, crop_size)).ravel() for img in imgs])

    @staticmethod
    def load_outlier_pcamodel() -> 'PCAModel':
        """
        :return: A pre-trained PCAModel object to detect outliers or abnormal CXR images.
        """
        data = res.resource_string(__name__, "model/pca-001.pkl")
        return pickle.loads(data)
