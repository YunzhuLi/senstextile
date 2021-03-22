import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def synchronize_touch_and_touch(touch_0_seq, touch_1_seq, touch_0_ts, touch_1_ts, offset):
    touch_0_idx = []
    touch_1_idx = []
    idx_0 = 0
    idx_1 = 0
    while touch_0_ts[idx_0] > touch_1_ts[idx_1]:
        idx_1 += 1

    for i in range(idx_1, len(touch_1_seq)):
        flag = False
        while touch_0_ts[idx_0] < touch_1_ts[i]:
            idx_0 += 1
            if idx_0 >= min(len(touch_0_seq), len(touch_0_seq) - offset):
                flag = True
                break
        if flag:
            break

        touch_0_idx.append(idx_0 + offset)
        touch_1_idx.append(i)

    touch_0_idx = np.stack(touch_0_idx)
    touch_1_idx = np.stack(touch_1_idx)

    return touch_0_idx, touch_1_idx


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size))
        return res


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.rcParams["figure.figsize"] = (16, 12)
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 25

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    ax.set_xlim([-0.5, cm.shape[1] - 0.5])
    ax.set_ylim([cm.shape[0] - 0.5, -0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


