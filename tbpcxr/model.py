

from sklearn import decomposition
import SimpleITK as sitk
from typing import List

import numpy as np

from sklearn.covariance import MinCovDet
from sklearn.covariance import EllipticEnvelope

from . import registration


class PCAModel:

    SAMPLE_IMAGE_SIZE = 64
    CROP_SIZE = 6

    def __init__(self):
        self.pca = None
        self.pca_cov = None
        self.outlier_detector = None
        self.image_atlas = None

        dim = 2
        self.image_ref = sitk.Image([self.SAMPLE_IMAGE_SIZE]*dim, sitk.sitkUInt8)
        self.image_ref.SetOrigin([-0.5 + (0.5 / self.SAMPLE_IMAGE_SIZE)]*dim)
        self.image_ref.SetSpacing([1.0 / self.SAMPLE_IMAGE_SIZE]*dim)
        pass

    def compute(self, arr, components):
        # arr is an array of ( n, feature_vector, )

        self.pca = decomposition.PCA(n_components=components)

        self.pca.fit(arr)

        reduced_cxr = self.pca.transform(arr)

        self.pca_cov = MinCovDet().fit(reduced_cxr)

        train_res = self.residuals(arr)
        train_dist = self.robust_distance(arr)

        X = np.stack((train_res, train_dist), axis=-1)

        self.outlier_detector = EllipticEnvelope(contamination=0.10)
        self.outlier_detector.fit(X)

    def register_to_atlas_and_resample(self, image: sitk.Image, verbose=0) \
            -> List[sitk.Image]:

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

    def outlier_predictor(self, arr):

        res = self.residuals(arr)
        dist = self.robust_distance(arr)

        X = np.stack((res, dist), axis=-1)
        return self.outlier_detector.predict(X)

    def residual_images(self, arr):
        reduced_cxr = self.pca.transform(arr)
        restored_cxr = self.pca.inverse_transform(reduced_cxr)
        residual = (restored_cxr - arr) ** 2
        return self._arr_to_images(residual)

    def restored_images(self, arr):
        reduced_cxr = self.pca.transform(arr)
        restored_cxr = self.pca.inverse_transform(reduced_cxr)
        return self._arr_to_images(restored_cxr)

    def residuals(self, arr):
        reduced_cxr = self.pca.transform(arr)
        restored_cxr = self.pca.inverse_transform(reduced_cxr)

        # Compute root mean squared of residual image vectors
        return np.sqrt(np.mean((restored_cxr - arr) ** 2, axis=1))

    def robust_distance(self, arr):
        """ """
        reduced_cxr = self.pca.transform(arr)

        return np.sqrt(self.pca_cov.mahalanobis(reduced_cxr))

    def avg(self):
        """Return the average image or first computed component"""
        comps = self.pca.components_
        comps = comps.reshape(comps.shape[0], self.SAMPLE_IMAGE_SIZE - 2 * self.CROP_SIZE,
                              self.SAMPLE_IMAGE_SIZE - 2 * self.CROP_SIZE)
        return sitk.Normalize(sitk.GetImageFromArray(comps[0, :, :]))

    def _arr_to_images(self, arr):
        number_of_images = arr.shape[0]
        return [sitk.GetImageFromArray(
            arr[i, :].reshape(self.SAMPLE_IMAGE_SIZE - 2 * self.CROP_SIZE,
                              self.SAMPLE_IMAGE_SIZE - 2 * self.CROP_SIZE))
            for i in range(number_of_images)]

    def _images_to_arr(self, imgs):

        return np.stack([sitk.GetArrayFromImage(sitk.Crop(img, [self.CROP_SIZE]*2, [self.CROP_SIZE]*2)).ravel() for img in imgs])
