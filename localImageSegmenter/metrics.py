from datetime import datetime
import numpy as np

def get_metrics_str(all_metrics, to_use=None):
    out_str = ""
    for name, val in all_metrics.items():
        if to_use is None or name in to_use:
            out_str += f" {name} {val:.4g}"
    return out_str

def get_metric_csv_row(metrics):
    now_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parts = [now_str, metrics['TP'], metrics['FP'], metrics['TN'],
             metrics['FN'], round(metrics['precision'], 4),
             round(metrics['recall'], 4), round(metrics['f1'], 4),
             metrics['defined'], metrics['duration']]
    return ','.join([str(p) for p in parts]) + '\n'


def get_metrics(tp, fp, tn, fn, defined_sum, duration, loss=float('nan')):
    total = (tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    assert not np.isnan(tp)
    assert not np.isnan(fp)
    assert not np.isnan(fn)
    assert not np.isnan(fp)
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        iou = tp / (tp + fp + fn)
    else:
        precision = recall = f1 = iou = float('NaN')
    return {
        "accuracy": accuracy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "true_mean": (tp + fn) / total,
        "pred_mean": (tp + fp) / total,
        "defined": defined_sum,
        "duration": duration,
        "loss": loss,
    }