import numpy as np
import scipy.misc
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure

from utils import *


predefined_threshold = 1.0


def true_positive(gt_image, predicted_image, threshold):
    """ Returns the amount of pixel which are positive in both images.

    :param gt_image: ground truth image
    :param predicted_image: predicted image
    :param threshold: threshold to make binary decision
    :return: amount of pixel which are positive in both images
    """
    return np.count_nonzero(gt_image[predicted_image >= threshold])


def true_negative(gt_image, predicted_image, threshold):
    """ Returns the amount of pixel which are negative in both images.

    :param gt_image: ground truth image
    :param predicted_image: predicted image
    :param threshold: threshold to make binary decision
    :return: amount of pixel which are negative in both images
    """
    return np.count_nonzero((gt_image < threshold)[(predicted_image < threshold)])


def false_positive(predicted_image, true_positive, threshold):
    """ Returns the amount of pixel which are negative in the ground truth but positive in predicted image.

    :param predicted_image: predicted image
    :param true_positive: amount of correct positive pixels
    :param threshold: threshold to make binary decision
    :return: amount of pixel which are negative in the ground truth but positive in prediction image
    """
    return abs(np.count_nonzero(predicted_image >= threshold) - true_positive)


def false_negative(gt_image, threshold, true_positive):
    """ Returns the amount of pixel which are positive in the ground truth but negative in predicted image.

    :param gt_image: ground truth image
    :param true_positive: amount of correct positive pixels
    :param threshold: threshold to make binary decision
    :return: amount of pixel which are positive in the ground truth but negative in predicted image
    """
    return abs(np.count_nonzero(gt_image > threshold) - true_positive) #FN


def example_metric(gt_image, predicted_image):
    """ Computes an example metric.

    :param gt_image: array - Ground truth of one tumor segment
    :param predicted_image: array - Predicted ground truth of one tumor segment
    :return: tuple(float, string) - example metric value, name of the metric
    """
    return np.abs(gt_image.sum() - predicted_image.sum()), "example"


def dice_score(gt_image, predicted_image, threshold=predefined_threshold):
    """ Computes the dice score for two binary images.
        TP*2 /(TP*2 + FP + FN)

    :param gt_image: array - Ground truth of one tumor segment
    :param predicted_image: array - Predicted ground truth of one tumor segment
    :param threshold: float - threshold where to seperate the 2 values
    :return: tuple(float, string) - dice score, name of the metric
    """
    tp = true_positive(gt_image,predicted_image,threshold)
    fp = false_positive(predicted_image, tp, threshold)
    fn = false_negative(gt_image, threshold, tp)

    # print("--------------------------------------------")
    # print("true positive %5.0f" % tp)
    # print("false positive %5.0f" % fp)
    # print("false negative %5.0f" % fn)

    tp += 1 # add 1 to avoid division by zero

    return tp * 2.0 / (tp * 2.0 + fn + fp), \
           "dice score"


def hausdorff_distance(gt_image, predicted_image):
    """
    Hausdorff Distance. http://pythonhosted.org/MedPy/_modules/medpy/metric/binary.html

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    input1 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Presumably does not influence the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```input1``` and the
        object(s) in ```input2```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    """
    hd1 = __surface_distances(gt_image, predicted_image, None, 1).max()
    hd2 = __surface_distances(predicted_image, gt_image, None, 1).max()
    return max(hd1, hd2), "hausdorff distance"


def sensitivity(gt_image, predicted_image, threshold=predefined_threshold):
    """ Computes the sensitivity score for two binary images. Measures the overlap.
        TP / (TP + FN)

    :param gt_image: array - Ground truth of one tumor segment
    :param predicted_image: array - Predicted ground truth of one tumor segment
    :param threshold: float - threshold for color value
    :return: tuple(float, string) - dice score, name of the metric
    """
    tp = true_positive(gt_image, predicted_image, threshold)
    fn = false_negative(gt_image, threshold, tp)

    # print("--------------------------------------------")
    # print("true positive %5.0f" % tp)
    # print("false negative %5.0f" % fn)

    tp += 1 # add 1 to avoid division by zero

    return tp * 2.0 / (tp * 2.0 + fn), \
           "sensitivity"


def specificity(gt_image, predicted_image, threshold=predefined_threshold):
    """ Computes the sensitivity score for two binary images. Its the counter part to sensitivity.
        TN / (TN + FP)

    :param gt_image: array - Ground truth of one tumor segment
    :param predicted_image: array - Predicted ground truth of one tumor segment
    :param threshold: float - threshold for color value
    :return: tuple(float, string) - dice score, name of the metric
    """
    tn = true_positive(gt_image, predicted_image, threshold)
    tp = true_positive(gt_image, predicted_image, threshold)
    fp = false_positive(gt_image, tp, threshold)

    # print("--------------------------------------------")
    # print("true negative %5.0f" % tn)
    # print("true positive %5.0f" % tp)
    # print("false positive %5.0f" % fp)

    tn += 1 # add 1 to avoid division by zero

    return tn / (tn + fp), \
           "specificity"


def get_img_row(overall_img, img_size, row, channels=10):
    """ Return all images of one slice of a brain.

    :param overall_img: array - the entire image
    :param img_size: integer - size of the image
    :param row: integer - row index
    :param channels: integer - images/channels for each row
    :return: array - all images of one slice of a brain
    """
    from_ = row * img_size
    to = from_ + img_size
    img_row = overall_img[from_:to]
    return np.array(np.split(img_row, channels, axis=1))


def prepare_images(file, batch_size=16, channels=10):
    """ Transforms the input file to an array with the shape [batch_size, dimensions, width, height].

    :param file: string - path to the file
    :param batch_size: integer - size/rows of a batch
    :param channels: integer - images/channels for each row/brain
    :return: array - prepared images with the shape [batch_size, dimensions, width, height]
    """
    input_img = np.array(scipy.misc.imread(file, flatten=True))  # (2048, 1280)
    img_size = int(input_img.shape[1] / channels)
    slices = int(input_img.shape[0] / img_size)
    output = np.empty((slices, channels, img_size, img_size))
    for i in range(batch_size):
        output[i] = get_img_row(input_img, img_size, i)
    return output


def execute_metrics(images, metrics=[example_metric], channels=3, gt_offset=0, predicted_offset=7):
    """ Execution functions to apply multiple metric functions on images.

    :param images: array - row of images to apply metric functions on
    :param metrics: array - metric functions to apply
    :param channels: integer - amount of channels to compare
    :param gt_offset - offset in the image row for the ground truth images
    :param predicted_offset - offset in the image row for the target images
    :return: tuple(array, array) - computed metric values (slices(rows), metrics, tumor regions), name of the metric
    """
    # For each image, metric and tumor region
    results = np.empty((images.shape[0], len(metrics), channels))
    metric_names = {}
    for i, row in enumerate(images):
        gt = row[gt_offset:channels]
        predicted = row[predicted_offset:predicted_offset + channels]
        for j, metric in enumerate(metrics):
            for k, gt_image, predicted_image in zip(range(3), gt, predicted):
                results[i, j, k], metric_names[j] = metric(gt_image, predicted_image)
        # TODO: Later plot the image with a table containing the metrics (for presentation)
    return results, metric_names


def __surface_distances(input1, input2, voxelspacing=None, connectivity=1.0):
    """
    http://pythonhosted.org/MedPy/_modules/medpy/metric/binary.html
    The distances between the surface voxel of binary objects in input1 and their
    nearest partner surface voxel of a binary object in input2.
    """
    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, input1.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(input1.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(input1):
        return np.array([float("inf")])  # Can´t calculate the hausdorff distance if there is no object
    if 0 == np.count_nonzero(input2):
        return np.array([float("inf")])  # Can´t calculate the hausdorff distance if there is no object

        # extract only 1-pixel border line of objects
    input1_border = input1 ^ binary_erosion(input1, structure=footprint, iterations=1)
    input2_border = input2 ^ binary_erosion(input2, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~input2_border, sampling=voxelspacing)
    sds = dt[input1_border]

    return sds


if __name__ == '__main__':
    images = prepare_images(file='test-x/test_0118.png')
    metrics = execute_metrics(images, [dice_score, specificity, sensitivity, hausdorff_distance])
    # slices(rows) x metrics x 3(tumor regions)
    print(metrics)
