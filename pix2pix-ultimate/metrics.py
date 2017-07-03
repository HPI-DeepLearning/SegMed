import numpy as np

def dice_score(source, target):

    k=5
    channels = 1
    # segmentation
    source = np.zeros((100,100), dtype='float')
    #source[30:70, 40:80] = 1

    # ground truth
    target = np.zeros((100,100), dtype='float')
    #target[30:70, 40:80] = 1

    sum = 0;
    for c in range(channels + 1):
        print(c)
        sum += np.sum(target[source==c])*2.0 / (np.sum(target) + np.sum(source))	
	
    return sum
    
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
    """Return all metric values as array
    1. dice_score
    2. hausdorf_distance
    3. sensitivity
    4. specificity"""

    return [dice_score(source, target),
        hausdorf_distance(source, target),
        sensitivity(source, target),
        specificity(source, target)]

def metrics_as_string(metrics):
    """Prints metrics with the correct name
    Values as Array with following order:
    1. dice_score
    2. hausdorf_distance
    3. sensitivity
    4. specificity
    """                

    numerals = 5
    decimal = 3
    print("Metrics: dice score: %{0}.{1}f, hausdorf_distance: %{0}.{1}f, \
sensitivity: %{0}.{1}f, specificity: %{0}.{1}f".format(numerals, decimal) \
        % (metrics[0], metrics[1], metrics[2], metrics[3]))