import os
from glob import glob
from utils import *


# Return all images of one slice of a brain (seperated)
def get_img_row(overall_img, img_size, i):
    from_ = i * img_size
    to = from_ + img_size
    img_row = overall_img[from_:to]
    return np.array(np.split(img_row, 10, axis=1))

# Output array has the shape [batch_size, dimensions, width, height]
def prepare_images(file, batch_size = 16):
    channels = 10  # 10 dimensions for each slice of a brain
    input_img = np.array(scipy.misc.imread(file, flatten=True))  # (2048, 1280)
    img_size = int(input_img.shape[1] / channels)
    slices = int(input_img.shape[0] / img_size)
    output = np.empty((slices, channels, img_size, img_size))
    for i in range(16):
        output[i] = get_img_row(input_img, img_size, i)
    return output

def example_metric(source, target):
    # Image of one tumor region
    return np.abs(source.sum() - target.sum())

def execute_metrics(images, metrics=[example_metric]):
    # For each image, metric and tumor region
    results = np.empty((images.shape[0], len(metrics), 3))
    for i, row in enumerate(images):
        gt = row[:3]
        predicted = row[7:]
        for j, metric in enumerate(metrics):
            for k, gt_region, pred_region in zip(range(3), gt, predicted):
                results[i, j, k] = metric(gt_region, pred_region)
    return results

if __name__ == '__main__':
    images = prepare_images(file='test-x/test_0118.png')
    metrics = execute_metrics(images)
    # slices(rows) x metrics x 3(tumor regions)
    print(metrics)
