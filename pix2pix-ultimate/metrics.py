def dice_score(source, target):
    #TODO
    return 0
    
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