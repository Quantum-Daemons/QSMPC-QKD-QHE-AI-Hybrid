# The medical CKKS student was a constant predictor, and the class labels were inverted
### Round 24, 2026-08-09

Two defects in the medical use case, found by following the metric-name mismatch the plan flagged as
a cosmetic registry inconsistency. It was not cosmetic.

## Defect 1: the model that runs under CKKS had learned nothing

`medical_fl.student` is the FHE-feasible CNN - the model the encrypted medical path actually
evaluates. Evaluated directly from its published `weights.json`, the way the browser evaluates it:

```
predictions over 156 test images:  {class 1: 156}      <- one class, every input
accuracy                           0.730769
majority-class rate                0.730769            <- exactly equal
auc                                0.354428            <- BELOW random
```

**It emitted the same class for every input.** Its accuracy of 0.7308 is precisely 114/156, the
majority-class rate, so accuracy was measuring the class balance rather than the model.

**Why it was invisible.** The registry declares `auc` as this model's headline; the record stored
`accuracy`. AUC would have shown 0.354 - below random, unmissable. Accuracy showed 0.7308 against a
published 0.901 and read as "behind, but in the right range". The mismatch was on the plan's list as
a tidy-up item; it was in fact the thing hiding the failure.

**Why it collapsed.** The student had neither of the two protections the teacher in the same file
already had, twenty lines above:

| | teacher | student (before) |
|---|---|---|
| class-weighted loss | yes | no |
| selection metric | AUC | **accuracy** |

`train_torch`'s own docstring says selecting on accuracy hides "the class that matters is not being
learned at all". At 2.71:1, plain cross-entropy is minimised by answering with the majority class,
and accuracy-based early stopping is maximised by it. Both signals pointed at the constant.

## Defect 2: the published class labels were inverted

MedMNIST ships the mapping in its own package metadata:

```
INFO["breastmnist"]["label"] == {'0': 'malignant', '1': 'normal, benign'}
```

Four places in this project asserted the opposite - `["benign/normal", "malignant"]`:
the published `weights.json`, the browser fixture in `prep_model_inputs.py`, the HF Space, and the
use-case sample selection (`X[y == 1][0]` commented "a malignant case", which selects a benign one).

`onnxInference.ts` renders `classes[predictedClass]`, so **the demo displayed the opposite
diagnosis to what the model meant.** Combined with defect 1, the medical use case reported
"malignant" for every patient while the model's output meant "normal, benign".

The project contradicted itself in-tree: `medical_breastmnist_resnet18/eval_sample.json` carried the
correct `{'0': 'malignant'}` while the student's carried the inverted one, because they are written
by different code paths. Two files, same repo, opposite claims about which class is cancer.

## What was fixed, and what it bought

Class-weighted loss + AUC selection + a tuned malignant cut folded into the dense head (argmax over
two logits IS a 0.5 threshold, and the CKKS path has nowhere downstream to apply a cut):

| | before | after |
|---|---|---|
| distinct classes predicted | **1** | 2 |
| AUC | 0.3544 (below random) | **0.6092** |
| malignant recall | 1 of 42 | **8 of 42** |
| accuracy | 0.7308 | 0.6859 |

Accuracy FELL, and that is the improvement: it fell because the model stopped answering with the
majority class. On an imbalanced screening task accuracy is maximised by not screening.

**A first attempt tuned the cut on minority-class F1 and degenerated**, picking the lowest grid
point and predicting malignant for all 156 images. That is not a bug in the folding - it is what F1
does when discrimination is weak, since recall goes to 1 while precision only falls to the base
rate. The validation score it chose, 0.4242, is exactly the all-positive value 2(0.269)/1.269.
Youden's J cannot be gamed that way (all-positive gives TPR=1, FPR=1, J=0), and a guard now rejects
any threshold that collapses to one class regardless of what it scored.

## The number is one draw, and that is measured

Six runs, identical config, seed the only difference:

| | mean | sd | range |
|---|---|---|---|
| teacher AUC | 0.8905 | 0.0198 | 0.8630 - 0.9216 |
| student AUC | 0.5788 | 0.0820 | 0.4906 - 0.7155 |

**The teacher straddles its own published baseline of 0.901 - 1 of 6 runs reaches it.** The
student's range runs from chance to 0.72. On a 156-image test split that is what a single recorded
AUC is worth, and the same caution the satellite investigation reached by a different route:
report the sample size and the spread, or the comparison to a published benchmark is theatre.

Two recorded teacher numbers moved on rerun with no functional change to the teacher path -
resnet18 0.9147 to 0.8779 and resnet50 0.7988 to 0.8876. The resnet50 figure now exceeds its
published 0.857 baseline. That is variance, not an improvement, and is recorded here so it is not
later cited as one.

## What is NOT claimed

The student is still weak: balanced accuracy 0.5294, malignant precision 0.348. A 3,226-parameter
one-conv-block CNN at 14x14 on 780 ultrasound images is not a diagnostic model, and no threshold
recovers what the capacity does not support. The honest summary is that it went from *fraudulently
respectable* to *visibly weak*, which is the direction that matters when the number is displayed
beside a published baseline.
