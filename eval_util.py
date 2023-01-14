import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_testing_results(truth, prediction, labels=None, target_names=None):

    # usage:
    # plot_testing_results(gt, model_pred, [0, 1, 2, 3, 4], ['nilm', 'asc_us', 'lsil', 'hsil', 'agc'])

    if labels is None:
        labels = [0, 1, 2, 3, 4, 5]
    if target_names is None:
        target_names = ['others', 'AUS', 'PTC', 'MTC', 'FN', 'SS']

    print(classification_report(truth, prediction, labels=labels, target_names=target_names))
    cm = confusion_matrix(truth, prediction, labels=labels)
    
    print("             Confusion Matrix:\n")
    print(cm)

    