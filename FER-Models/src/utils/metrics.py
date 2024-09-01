'''
모델 평가에 사용되는 지표 계산 및 시각화 함수를 제공
'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import balanced_accuracy_score


def accuracy_(predicts, targets):
    """
    accuracy를 계산
    """
    correct_num = torch.eq(predicts, targets).sum().item()
    total_num = targets.numel()  # 대상이 tensor이든 batch of tensor이든 호환성을 보장함
    accuracy = correct_num / total_num
    return accuracy


def balanced_accuracy_(predicts, targets):
    """
    balanced accuracy를 계산
    """
    predicts = predicts.cpu().numpy() if not isinstance(predicts, np.ndarray) else predicts
    targets = targets.cpu().numpy() if not isinstance(targets, np.ndarray) else targets
    balanced_acc = balanced_accuracy_score(targets, predicts)
    return np.around(balanced_acc, decimals=4)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    """
    표정별 confusion matrix를 시각화
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, format='png')
    plt.show()



"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'accuracy': accuracy_,
    'balanced accuracy': balanced_accuracy_,
}