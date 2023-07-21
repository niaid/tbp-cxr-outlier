
Example
=======

To classify a single image:

.. code-block:: python

 from tbpcxr.model import Model
 from tbpcxr.utilities import read_dcm

 outlier_model = Model.load_outlier_pcamodel()

 img = read_dcm(path_to_file)

 arr = outlier_model.to_observations([img])

 if outlier_model.outlier_predictor(arr)[0] == -1:
    print("{} is an outlier".format(path_to_file))


Multiple images can efficiently be processed by using Python's `map` function, which

.. code-block:: python

 from tbpcxr.model import Model
 from tbpcxr.utilities import read_dcm

 outlier_model = Model.load_outlier_pcamodel()

 arr = outlier_model.to_observations(map(read_dcm, image_file_list))

 results = outlier_model.outlier_predictor(arr)

 for fn in [fn for fn, o in zip(image_file_list, results) if o == -1]:
    print("{} is an outlier".format(fn))
