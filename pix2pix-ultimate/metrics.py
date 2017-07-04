import numpy as np
import scipy.misc

from utils import *


def example_metric(source, target):
    """ Computes an example metric

    :param source: array - Ground truth of one tumor segment
    :param target: array - Predicted ground truth of one tumor segment
    :return: float - example metric value
    """
    return np.abs(source.sum() - target.sum())


def dice_score(source, target, channel_value=0):
    """ Computes the dice score for two images for a given value

    :param source: array - Ground truth of one tumor segment
    :param target: array - Predicted ground truth of one tumor segment
    :param channel_value: float - value to compute the dice score for
    :return: float - dice score
    """
    return np.sum(target[source == channel_value])*2.0 / (np.sum(target) + np.sum(source))


def hausdorf_distance(source, target):
    #TODO
    return 0


def sensitivity(source, target):
    #TODO
    return 0


def specificity(source, target):
    #TODO
    return 0

    
def compute_metrics(source, target):
    """ Return all metric values as array
    1. dice_score
    2. hausdorf_distance
    3. sensitivity
    4. specificity"""

    return [dice_score(source, target),
        hausdorf_distance(source, target),
        sensitivity(source, target),
        specificity(source, target)]


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


def execute_metrics(images, metrics=[example_metric],channels=3 ,source_offset=0, target_offset=7):
    """ Execution functions to apply multiple metric functions on images.

    :param images: array - row of images to apply metric functions on
    :param metrics: array - metric functions to apply
    :param channels: integer - amount of channels to compare
    :param source_offset - offset in the image row for the source images
    :param target_offset - offset in the image row for the target images
    :return: array - all computed metric values
    """
    # For each image, metric and tumor region
    results = np.empty((images.shape[0], len(metrics), channels))
    for i, row in enumerate(images):
        gt = row[source_offset:channels]
        predicted = row[target_offset:channels]
        for j, metric in enumerate(metrics):
            for k, source_image, target_image in zip(range(channels), gt, predicted):
                results[i, j, k] = metric(source_image, target_image)
        # TODO: Later plot the image with a table containing the metrics (for presentation)
    return results


if __name__ == '__main__':
    images = prepare_images(file='test-x/test_0118.png')
    metrics = execute_metrics(images, [dice_score])
    # slices(rows) x metrics x 3(tumor regions)
    print(metrics)

