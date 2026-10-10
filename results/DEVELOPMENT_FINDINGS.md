# First development findings — 10 October 2026

## Dataset audit
Original: 3,349/120/80 images in train/validation/test. After transitive source-name
and byte-hash duplicate quarantine: 3,313/119/80. 37 copies were excluded rather than
moving held-out data into training. Source grouping still cannot guarantee independent
trays because source filenames identify video frames.

The unused category ID 0 was not treated as a real sixth category. Actual annotations
use loose-meat, packaging-error, twisted-meat, unsealed and wrinkle. Wrinkle has only
18 training boxes and zero held-out boxes; generalisation for this class is unmeasured.

## Regression count baseline
1,073 unique training source images and 119 validation source images were used.
We tested mean, median, linear, three tree depths, three SVR C values, and one neural
network configuration under both 2x2 and 4x4 feature grids. Counts are annotation-box
counts; they are not rasher counts or certified inspection labels.

| Predictor | Grid | Validation MAE (defects/image) |
|---|---|---:|
| Constant median (always 1) | irrelevant | 0.2437 |
| Best learned feature model: RBF SVR, C=0.1 | 4x4 | 0.2982 |
| Constant training mean | irrelevant | 0.3420 |

The learned feature models do **not** beat the constant-median baseline on MAE.
This is a negative baseline result, not a successful inspection system. A low absolute
MAE alone is misleading here because many frames have exactly one labelled defect.
The next meaningful comparison is whether localisation-based detection improves on
this common count task while also supplying correct locations/classes.

All scores are development validation results. Test predictions have not been produced.
No detector accuracy or laptop runtime measurements exist yet.
