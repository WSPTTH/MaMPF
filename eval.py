import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json

_EPSILON = 1e-20


def _confusion_matrix(real, pred, label_id):
    real_app = real == label_id
    pred_app = pred == label_id
    tp_num = np.sum(np.logical_and(real_app, pred_app))
    fn_num = np.sum(real_app) - tp_num
    fp_num = np.sum(pred_app) - tp_num
    tn_num = len(real) - tp_num - fn_num - fp_num
    return tp_num, tn_num, fp_num, fn_num


def _evaluate_fpr_and_tpr(real, pred, class_num):
    tp, tn, fp, fn = 0, 0, 0, 0
    tpr, fpr, precision, f1 = {}, {}, {}, {}
    for app_ind in range(class_num):
        tp_app, tn_app, fp_app, fn_app = _confusion_matrix(real, pred, app_ind)
        tpr[str(app_ind)] = float(tp_app / (tp_app + fn_app + _EPSILON))
        fpr[str(app_ind)] = float(fp_app / (fp_app + tn_app + _EPSILON))
        precision[str(app_ind)] = float(tp_app / (tp_app + fp_app + _EPSILON))
        f1[str(app_ind)] = float(2 * tpr[str(app_ind)] * precision[str(app_ind)] /
                                 (tpr[str(app_ind)] + precision[str(app_ind)] + _EPSILON))

        tp += tp_app
        tn += tn_app
        fp += fp_app
        fn += fn_app

    for dd in (tpr, fpr, precision, f1):
        dd['macro'] = float(sum(dd.values()) / (len(dd) + _EPSILON))

    tpr['micro'] = float(tp / (tp + fn + _EPSILON))
    fpr['micro'] = float(fp / (fp + tn + _EPSILON))
    precision['micro'] = float(tp / (tp + fp + _EPSILON))
    f1['micro'] = float(2 * tpr['micro'] * precision['micro'] / (tpr['micro'] + precision['micro'] + _EPSILON))

    return tpr, fpr, precision, f1


def _evaluate_ftf(tpr, fpr, real, total_class):
    class_num = np.zeros(total_class, dtype=np.float)
    for ix in range(total_class):
        class_num[ix] = (real == ix).astype(np.float).sum()
    weight = class_num / class_num.sum()
    res = 0.0
    for key in tpr:
        if key in ('macro', 'micro'):
            continue
        res += weight[int(key)] * tpr[key] / (1.0 + fpr[key])
    return float(res)


def _one_hot(index, class_num):
    res = np.eye(class_num)[np.array(index).reshape(-1)]
    return res


def evaluation_metric(real, pred, class_num):
    real = np.array(real)
    pred = np.array(pred)
    tpr, fpr, precision, f1 = _evaluate_fpr_and_tpr(real, pred, class_num)
    ftf = _evaluate_ftf(tpr, fpr, real, class_num)
    acc = float(accuracy_score(real, pred))

    res = {
        'ACC': acc, 'FTF': ftf, 'TPR': tpr, 'FPR': fpr, 'Precision': precision, 'F1': f1,
        'AVE-TPR': tpr['micro'], 'AVE-FPR': fpr['micro']
    }
    return res


def save_res(res, filename):
    with open(filename, 'w') as fp:
        json.dump(res, fp, indent=1, sort_keys=True)


def evaluate(real, pred):
    class_num = len(real)
    real = np.concatenate(real)
    pred = np.concatenate(pred)
    res = evaluation_metric(real, pred, class_num)
    return res
