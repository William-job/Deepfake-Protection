import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x).flatten()


def _validate_inputs(preds, labels):
    preds = to_numpy(preds)
    labels = to_numpy(labels)
    valid_mask = ~(np.isnan(preds) | np.isnan(labels))
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    # Ensure labels are integers (0 or 1)
    labels = labels.astype(int)
    return preds, labels


def find_optimal_threshold(preds, labels):
    preds, labels = _validate_inputs(preds, labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, preds)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return float(thresholds[best_idx])


def compute_eer(preds, labels):
    preds, labels = _validate_inputs(preds, labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, preds)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[eer_idx])


def compute_tpr_at_fpr(preds, labels, target_fpr):
    preds, labels = _validate_inputs(preds, labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(labels, preds)
    idx = np.searchsorted(fpr, target_fpr, side='right') - 1
    idx = max(0, min(idx, len(fpr) - 1))
    return float(tpr[idx])


def compute_accuracy(preds, labels, threshold=0.5):
    preds, labels = _validate_inputs(preds, labels)
    preds_binary = (preds >= threshold).astype(int)
    return accuracy_score(labels, preds_binary)


def compute_auc(preds, labels):
    preds, labels = _validate_inputs(preds, labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


def compute_ap(preds, labels):
    preds, labels = _validate_inputs(preds, labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    return average_precision_score(labels, preds)


def compute_f1(preds, labels, threshold=0.5):
    preds, labels = _validate_inputs(preds, labels)
    if len(labels) == 0:
        return 0.0
    preds_binary = (preds >= threshold).astype(int)
    return f1_score(labels, preds_binary)


def compute_all_metrics(preds, labels, threshold=None):
    # 阈值协议：threshold=None 时在传入样本上 Youden 选阈值（会高估 acc/f1/eer）。
    # 严谨评估应先在 val 集确定 threshold，再冻结传入本函数用于 test 集。
    if threshold is None:
        threshold = find_optimal_threshold(preds, labels)
    return {
        "accuracy": compute_accuracy(preds, labels, threshold),
        "auc": compute_auc(preds, labels),
        "ap": compute_ap(preds, labels),
        "f1": compute_f1(preds, labels, threshold),
        "threshold": threshold,
        "eer": compute_eer(preds, labels),
        "tpr_at_fpr_1": compute_tpr_at_fpr(preds, labels, 0.01),
        "tpr_at_fpr_01": compute_tpr_at_fpr(preds, labels, 0.001),
    }
