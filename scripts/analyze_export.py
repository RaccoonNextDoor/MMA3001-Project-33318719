"""Score exported custom-dataset YOLO JSON against the prepared manifest.

Validation-only threshold selection; no model inference or test-set evaluation.
JSON categories are one-based in Ultralytics 8.4.175 custom dataset exports.
Matching is confidence-ordered, class-aware, one-to-one, at IoU >= 0.5.
These pooled operating-point scores are distinct from Ultralytics macro AP.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    """Intersection over union for pixel [left, top, width, height] boxes."""
    x = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    union = a[2]*a[3]+b[2]*b[3]-x*y
    return x*y/union if union > 0 else 0.0


def score(records, predictions, threshold):
    """Return pooled/per-class detection scores and image-level count errors."""
    counts = [[0, 0, 0] for _ in range(5)]  # TP, FP, FN
    images = []
    for rec in records:
        truth = rec['boxes']
        pred = sorted((p for p in predictions[rec['file_name']] if p['score'] >= threshold),
                      key=lambda p: -p['score'])
        used = set()
        tp = 0
        for p in pred:
            cls = p['category_id']-1
            candidates = [(iou(p['bbox'], b['bbox']), j) for j, b in enumerate(truth)
                          if j not in used and b['class_id'] == cls]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= .5:
                used.add(index)
                counts[cls][0] += 1
                tp += 1
            else:
                counts[cls][1] += 1
        for j, b in enumerate(truth):
            if j not in used:
                counts[b['class_id']][2] += 1
        images.append(dict(file_name=rec['file_name'], true_count=len(truth),
                           predicted_count=len(pred), tp=tp, fp=len(pred)-tp,
                           fn=len(truth)-tp))
    def rates(c):
        t, f, n = c
        return dict(tp=t, fp=f, fn=n, precision=t/(t+f) if t+f else 0,
                    recall=t/(t+n) if t+n else None,
                    f1=2*t/(2*t+f+n) if 2*t+f+n else 0)
    result = rates([sum(c[k] for c in counts) for k in range(3)])
    errors = [r['predicted_count']-r['true_count'] for r in images]
    result.update(threshold=threshold, count_mae=sum(abs(e) for e in errors)/len(errors),
                  count_rmse=(sum(e*e for e in errors)/len(errors))**.5,
                  exact_count_images=sum(e == 0 for e in errors),
                  flagged_empty_images=sum(r['true_count']==0 and r['predicted_count']>0 for r in images),
                  missed_positive_images=sum(r['true_count']>0 and r['predicted_count']==0 for r in images))
    return result, [rates(c) for c in counts], images


def main():
    """Evaluate two exports on a fixed 0.05..0.95 confidence grid."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    records = [r for r in json.loads(args.manifest.read_text()) if r['split']=='valid']
    names = {r['file_name'] for r in records}
    assert len(names)==len(records)
    args.out.mkdir(parents=True, exist_ok=True)
    summary, sweep = {}, []
    for size in (320, 640):
        folder = args.exports/f'validation_{size}'
        pred = defaultdict(list)
        for row in json.loads((folder/'predictions.json').read_text()):
            assert row['file_name'] in names, 'Export contains images outside validation'
            assert 1 <= row['category_id'] <= 5
            pred[row['file_name']].append(row)
        rows = [score(records, pred, k/100)[0] for k in range(5, 100, 5)]
        best = max(rows, key=lambda r: (r['f1'], r['threshold']))
        metrics, classes, images = score(records, pred, best['threshold'])
        for row in rows:
            sweep.append(dict(model=f'yolov8n_{size}', **row))
        summary[str(size)] = dict(operating_point=metrics, classes=classes,
            ultralytics=json.loads((folder/'metrics.json').read_text()),
            at_025=score(records, pred, .25)[0])
        with (args.out/f'images_{size}.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(images[0]));writer.writeheader();writer.writerows(images)
    (args.out/'operating_points.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.out/'threshold_sweep.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(sweep[0]));writer.writeheader();writer.writerows(sweep)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
