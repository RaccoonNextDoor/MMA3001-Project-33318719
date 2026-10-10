"""Analytical checks for exported detection matching and counting."""
from collections import defaultdict
from scripts.analyze_export import iou, score


def test_iou_known_overlap():
    assert iou([0, 0, 2, 2], [1, 0, 2, 2]) == 1/3
    assert iou([0, 0, 1, 1], [2, 2, 1, 1]) == 0


def test_duplicate_and_wrong_class_are_false_positives():
    records = [dict(file_name='x.jpg', boxes=[dict(class_id=0, bbox=[0,0,10,10])])]
    predictions = defaultdict(list, {'x.jpg': [
        dict(category_id=2, bbox=[0,0,10,10], score=.99),
        dict(category_id=1, bbox=[0,0,10,10], score=.9),
        dict(category_id=1, bbox=[0,0,10,10], score=.8)]})
    metrics, classes, images = score(records, predictions, .5)
    assert (metrics['tp'], metrics['fp'], metrics['fn']) == (1,2,0)
    assert metrics['precision'] == 1/3
    assert metrics['count_mae'] == 2
    assert classes[1]['fp'] == 1


def test_empty_predictions_and_threshold_boundary():
    records = [dict(file_name='empty.jpg', boxes=[]),
               dict(file_name='positive.jpg', boxes=[dict(class_id=0, bbox=[0,0,10,10])])]
    pred = defaultdict(list, {'empty.jpg': [dict(category_id=1,bbox=[0,0,10,10],score=.5)]})
    metrics, _, _ = score(records, pred, .5)
    assert (metrics['tp'],metrics['fp'],metrics['fn']) == (0,1,1)
    assert metrics['flagged_empty_images'] == 1
    assert metrics['missed_positive_images'] == 1
    metrics, _, _ = score(records, pred, .55)
    assert metrics['fp'] == 0
