import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def cal_best_PRF(y_true,probas_pred):
    '''
    计算在任何阈值下，最好的precision，recall。f1
    :param y_true:
    :param probas_pred:
    :return:
    '''
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, probas_pred)

    f1s=(2*precisions*recalls)/(precisions+recalls)
    f1s[np.isnan(f1s)] = 0

    best_index=np.argmax(f1s)

    aupr = average_precision_score(y_true, probas_pred)

    return precisions[best_index],recalls[best_index],f1s[best_index],aupr

    # return precisions, recalls, f1s, thresholds