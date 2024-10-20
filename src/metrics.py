def dice_overlap(pred, target):
    # Dice: (2 x (A*B) / (A + B))
    intersection = (pred * target).sum()
    cardinalities = pred.sum() + target.sum()
    return 2 * intersection / cardinalities

def intersection_over_union(pred, target):
    # IOU : (A * B) / (A + B) - (A * B)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / union

def accuracy(pred, target):
    # Accuracy: correct / total
    correct = (pred == target).sum()
    return correct / pred.numel()

def sensitivity(pred, target):
    # Sensitivity: TP / (TP + FN)
    true_positives = (pred * target).sum()
    false_negatives = ((1 - pred) * target).sum()
    return true_positives / (true_positives + false_negatives)

def specificity(pred, target):
    # Specificity: TN / (TN + FP)
    true_negatives = ((1 - pred) * (1 - target)).sum()
    false_positives = (pred * (1 - target)).sum()
    return true_negatives / (true_negatives + false_positives)
