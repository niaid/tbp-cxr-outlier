import pkg_resources as res
import os

from sklearn import decomposition
import SimpleITK as sitk
from typing import List

import numpy as np

from abc import ABC, abstractmethod

from sklearn.covariance import MinCovDet
from sklearn.covariance import EllipticEnvelope

import pickle

from . import registration
from . import utilities


class Model(ABC):
    """
    This class is a base class for models of image representations.

    The images are registered to an atlas and resampled onto a reference image to make each image a comparable
    observation vector.

    Four states of the image data need to be defined:
     * normalized image: An sitk.Image which has been registered to the :attr:`image_atlas`, resampled onto the space
       of the :attr:`image_reference` then the intensities normalized.
     * observation vector : A normalized image converted into an array ( np.ndarray ).
     * feature vector: A representation of an image that has been converted into an observation then further reduced.

    """

    def __init__(self, reference_size=64, reference_crop=6):

        # check for sanity of remaining image
        assert reference_size > reference_crop * 2

        self.reference_crop = reference_crop
        """The reference image, image atlas, and input images are cropped by this amount on each edge for registration, and
        the PCA feature vector."""

        self.image_atlas = None
        """The atlas image which input images are registered too.

        This image must have the same physical extent as image_reference

        :type: sitk.Image or None
        """

        dim = 2
        self.image_reference = sitk.Image([reference_size] * dim, sitk.sitkUInt8)
        self.image_reference.SetOrigin([-0.5 + (0.5 / reference_size)] * dim)
        self.image_reference.SetSpacing([1.0 / reference_size] * dim)

    @abstractmethod
    def compute(self, observation_arr: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    @property
    def reference_size(self):
        """The reference image is defined as this number of pixels in each dimension."""
        return self.image_reference.GetSize()[0]

    def to_observations(self, images: List[sitk.Image], verbose: int = 0) -> np.ndarray:

        img_list = []
        for img in images:
            img = utilities.normalize_img(img, self.reference_size)

            img_list.append(self.register_to_atlas_and_resample(img, verbose=verbose))

        return self._images_to_arr(img_list)

    def register_to_atlas_and_resample(self, image: sitk.Image, verbose: int = 0) \
            -> sitk.Image:
        """
        Register to the input images to the atlas image, and resample to the reference image coordinates.

        :param image:
        :param verbose:
        :return: The registered and resampled image.
        """

        fixed_crop = self.image_atlas_cropped

        try:
            transform, metric_value = registration.cxr_affine(fixed_crop, moving=image, verbose=verbose)
        except RuntimeError as e:
            print("Registration Error:")
            print(e)
            transform = sitk.TranslationTransform(2)

        return sitk.Resample(image,
                             referenceImage=self.image_reference,
                             transform=transform,
                             outputPixelType=sitk.sitkFloat32)

    @property
    def image_atlas_cropped(self):
        crop_size = [int(s * self.reference_crop / self.reference_size) for s in self.image_atlas.GetSize()]
        return sitk.Crop(self.image_atlas, crop_size, crop_size)

    def _arr_to_images(self, observation_arr: np.ndarray) -> List[sitk.Image]:
        """

        :param observation_arr:
        :return:
        """
        size = self.reference_size - 2 * self.reference_crop
        if len(observation_arr) == size**2:
            return sitk.GetImageFromArray(observation_arr.reshape(size, size))

        return [sitk.GetImageFromArray(arr.reshape(size, size))
                for arr in observation_arr]

    def _images_to_arr(self, imgs: List[sitk.Image]) -> np.ndarray:
        crop_size = [self.reference_crop] * 2
        return np.stack([sitk.GetArrayFromImage(sitk.Crop(img, crop_size, crop_size)).ravel() for img in imgs])

    @staticmethod
    def load_outlier_pcamodel() -> 'Model':
        """
        :return: A pre-trained PCAModel object to detect outliers or abnormal CXR images.
        """
        return __class__.load_model("pca-35-06c")

    @staticmethod
    def load_model(name: str) -> 'Model':
        """
        Loads a model named in :data:`tbpcxr.model_list` or a provided filename of a pickled Model object.

        This method provides a unified interface for loading Model objects.

        :param name: The name of a model from the tbpcxr module or a path of a `pkl` file.
        :return: A restored :obj:`Model` object.
        """

        model_list = res.resource_listdir(__name__, "model")

        model_list = [os.path.splitext(fn)[0] for fn in model_list]
        if name in model_list:
            data = res.resource_string(__name__, os.path.join("model", name+".pkl"))
            return pickle.loads(data)
        else:
            with open(name, "rb") as fp:
                return pickle.load(fp)


class PCAModel(Model):
    """
    The PCAModel uses Principle Components Analyst to represent a feature space of images.

    This class represents a model of an expected image to detect if another image conforms to the model or is an
    outlier.
    """

    def __init__(self, reference_size=64, reference_crop=6):

        super().__init__(reference_size, reference_crop)

        self.pca = None
        """The PCA decomposition of the cropped input image as a vector."""

        self.pca_cov = None
        """The object used to estimate the covariance of the training data in the PCA feature space."""

        self.outlier_detector = None
        """The trained object used to classify a fitted image as a PCA feature vector."""

    def compute(self, observation_arr: np.ndarray, **kwargs) -> None:
        """
        Computes the PCA decomposition, and outlier classifier from the input feature array.

        :param observation_arr: A set of images converted to an array of image observation vectors.
        :param components:
        :param contamination:
        :return:
        """

        return self.__compute(observation_arr, **kwargs)

    def __compute(self, observation_arr: np.ndarray, *, components: int, contamination=0.10) -> None:

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
