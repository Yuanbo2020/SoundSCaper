import os
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def scale(x, mean, std):
    return (x - mean) / std


# ----------------------------------------------------------------------------------------------------------------------

from sklearn import metrics
def cal_acc_auc(predictions, targets):
    tagging_truth_label_matrix = targets
    pre_tagging_label_matrix = predictions

    # overall
    tp = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix > 1.5)
    fn = np.sum(tagging_truth_label_matrix - pre_tagging_label_matrix > 0.5)
    fp = np.sum(pre_tagging_label_matrix - tagging_truth_label_matrix > 0.5)
    tn = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix < 0.5)

    Acc = (tp + tn) / (tp + tn + fp + fn)

    aucs = []
    for i in range(targets.shape[0]):
        test_y_auc, pred_auc = targets[i, :], predictions[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)
    return Acc, final_auc





