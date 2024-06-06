import os
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):
    return (x - mean) / std


# ------------------ demo ---------------------------------------------------------------------------------------------
def calculate_scalar_demo(x):
    # print(x)
    # print(x.shape)
    assert True not in np.isnan(x)

    # for each in range(len(x)):
    #     row = x[each]
    #     # print(row)
    #     # print(row.shape)
    #     assert True not in np.isnan(row)

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std

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





