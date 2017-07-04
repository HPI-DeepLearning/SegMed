import numpy as np
import scipy.misc

from utils import *


def example_metric(gt_image, predicted_image):
    """ Computes an example metric

    :param gt_image: array - Ground truth of one tumor segment
    :param predicted_image: array - Predicted ground truth of one tumor segment
    :return: tuple(float, string) - example metric value, name of the metric
    """
    return np.abs(gt_image.sum() - predicted_image.sum()), "example"


def dice_score(gt_image, predicted_image, threshold_value=128.0):
    """ Computes the dice score for two images for a given value

    :param gt_image: array - Ground truth of one tumor segment
    :param predicted_image: array - Predicted ground truth of one tumor segment
    :param threshold_value: float - threshold for color value
    :return: tuple(float, string) - dice score, name of the metric
    """
    true_positive = np.count_nonzero(gt_image[predicted_image >= threshold_value])
    false_positive = abs(np.count_nonzero(predicted_image) - true_positive)
    false_negative = abs(np.count_nonzero(gt_image) - true_positive)

    # print("--------------------------------------------")
    # print("true positive %5.0f" % true_positive)
    # print("false positive %5.0f" % false_positive)
    # print("false negative %5.0f" % false_negative)

    true_positive += 1 # add 1 to avoid division by zero

    return true_positive * 2.0 / (true_positive * 2.0 + false_negative + false_positive), \
           "dice score"


def hausdorf_distance(gt_image, predicted_image):
    #TODO
    return 0


def sensitivity(gt_image, predicted_image):
    #TODO
    return 0


def specificity(gt_image, predicted_image):
    #TODO
    return 0

    
def compute_metrics(gt_image, predicted_image):
    """ Return all metric values as array
    1. dice_score
    2. hausdorf_distance
    3. sensitivity
    4. specificity"""

    return [dice_score(gt_image, predicted_image),
        hausdorf_distance(gt_image, predicted_image),
        sensitivity(gt_image, predicted_image),
        specificity(gt_image, predicted_image)]


def metrics_as_string(metrics):
    """ Prints metrics with the correct name
    Values as Array with following order:
    1. dice_score
    2. hausdorf_distance
    3. sensitivity
    4. specificity

    :param metrics: computed metric values as array
    :return: -
    """

    numerals = 5
    decimal = 3
    print("Metrics: dice score: %{0}.{1}f, hausdorf_distance: %{0}.{1}f, \
sensitivity: %{0}.{1}f, specificity: %{0}.{1}f".format(numerals, decimal) \
        % (metrics[0], metrics[1], metrics[2], metrics[3]))


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


def execute_metrics(images, metrics=[example_metric],channels=3 ,source_offset=0, predicted_offset=7):
    """ Execution functions to apply multiple metric functions on images.

    :param images: array - row of images to apply metric functions on
    :param metrics: array - metric functions to apply
    :param channels: integer - amount of channels to compare
    :param source_offset - offset in the image row for the ground truth images
    :param predicted_offset - offset in the image row for the target images
    :return: tuple(array, array) - computed metric values (slices(rows), metrics, tumor regions), name of the metric
    """
    # For each image, metric and tumor region
    results = np.empty((images.shape[0], len(metrics), channels))
    metric_names = {}
    for i, row in enumerate(images):
        gt = row[source_offset:channels]
        predicted = row[predicted_offset:predicted_offset + channels]
        for j, metric in enumerate(metrics):
            for k, gt_image, predicted_image in zip(range(3), gt, predicted):
                results[i, j, k], metric_names[j] = metric(gt_image, predicted_image)
        # TODO: Later plot the image with a table containing the metrics (for presentation)
    return results, metric_names


if __name__ == '__main__':
    images = prepare_images(file='test-x/test_0118.png')
    metrics = execute_metrics(images, [dice_score])
    # slices(rows) x metrics x 3(tumor regions)
    print(metrics)

