"""Evaluate the frozen detector on exported test predictions without tuning."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from scripts.analyze_export import score


def main():
    """Verify the checkpoint hash and apply the recorded confidence threshold."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--frozen', type=Path, default=Path('results/detector_validation/frozen_selection.json'))
    p.add_argument('--out', type=Path, default=Path('results/final_test'))
    args = p.parse_args()
    frozen = json.loads(args.frozen.read_text())
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if digest != frozen['weights_sha256']:
        raise ValueError('Checkpoint differs from frozen selection')
    records = [r for r in json.loads(args.manifest.read_text()) if r['split']=='test']
    names = {r['file_name'] for r in records}
    if len(names) != len(records) or not records:
        raise ValueError('Test manifest must contain unique filenames')
    predictions = defaultdict(list)
    for row in json.loads((args.export/'predictions.json').read_text()):
        if row['file_name'] not in names or row['category_id'] not in range(1, 6):
            raise ValueError('Unexpected image or class in test export')
        predictions[row['file_name']].append(row)
    metrics, classes, images = score(records, predictions, frozen['confidence'])
    # The validation-selected training-median baseline was frozen at one defect.
    errors = [1-len(r['boxes']) for r in records]
    baseline = dict(predicted_count=1, mae=sum(abs(e) for e in errors)/len(errors),
                    rmse=(sum(e*e for e in errors)/len(errors))**.5)
    summary = dict(status='FINAL TEST - no threshold search performed',
        images=len(records), annotations=sum(len(r['boxes']) for r in records),
        weights_sha256=digest, frozen_selection=frozen, operating_point=metrics,
        classes=dict(zip(['loose-meat','packaging-error','twisted-meat','unsealed','wrinkle'],classes)),
        count_baseline=baseline,
        ultralytics=json.loads((args.export/'metrics.json').read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.out/'image_results.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(images)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
