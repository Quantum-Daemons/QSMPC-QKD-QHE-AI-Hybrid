# The QCNN against published numbers on the same dataset (Round 37, 2026-08-13)

The browser verification checklist declined this check, with a reason: *"it is parity against an
independent reimplementation, not against a paper's numbers, because no published QCNN reports this
circuit on this data."*

**The reason is true of the circuit and false of the data.** The medical QCNN trains on
`sklearn.load_breast_cancer` - WDBC, UCI dataset 17 - which is one of the most heavily benchmarked
tabular datasets in existence, and quantum-model results on it are published. The decline was
therefore unfalsifiable in the wrong direction: it named a condition nobody had tested.

## The comparator, resolved to the primary source

**Quantum classical hybrid convolutional neural networks for breast cancer diagnosis**, Scientific
Reports 14 (2024), **DOI 10.1038/s41598-024-74778-7**, PMID 39433779, PMC11494181, open access.

Fetched and read rather than taken from a search summary, and that mattered: a search engine reported
this paper's WDBC figure as **97.7 %**, which is the *training* accuracy 97.099 % misread. The paper's
validation figure is **96.155 %**. One number would have entered this repository wrong.

| | features | split | quantum | logistic regression |
|---|---|---|---|---|
| Sci Rep 2024, WDBC | **30** | 70 / 30, **no test set** | QCCNN **0.96155** | **0.91184** |
| this project, medical QCNN | **8** | 70 / 15 / 15, held-out test | median **0.9070** over 5 seeds | median **0.9651** |

Both quantum figures are accuracy. The project's five seeds, measured for this document by re-running
`scripts/train/qcnn_variational.py --usecase medical --layers 3 --epochs 120` at seeds 0-4 and
restoring the shipped artifact afterwards:

| seed | QCNN accuracy | LR accuracy |
|---|---|---|
| **0 (shipped)** | **0.7326** | 0.9535 |
| 1 | 0.9186 | 0.9651 |
| 2 | 0.9070 | 0.9651 |
| 3 | 0.8023 | 0.9070 |
| 4 | 0.9535 | 0.9767 |

## Three differences that have to travel with the numbers

Reporting `0.9070` beside `0.96155` without them would be the `B2` violation this checklist already
names - *"the cited row scores the SAME TARGET the project scores"*.

1. **Feature count: 8 against 30.** Not a shortcut. The circuit is 8 qubits with one feature per
   wire, and the qubit count is bounded by what a state-vector simulation in a browser tab can carry.
   The paper's model sees the whole feature set.
2. **Split: a held-out test set against none.** The paper reports the *validation* split it selected
   on; this project reports a test split that neither the fit nor the checkpoint choice has seen. The
   comparison therefore flatters the paper, by an amount neither side measures.
3. **Architecture.** The paper's quantum layer sits inside a classical CNN; this circuit is a flat
   variational RY stack with a CNOT ring and no classical front-end. The paper does not state its
   qubit count, depth or encoding, so the two circuits cannot be aligned even in principle.

## What the comparison actually shows

**The paper's classical baseline is weak for this dataset, and that is where its margin comes from.**
This project's plain logistic regression on **8** features scores **0.9651**, above the paper's
QCCNN on **30** (0.96155) and well above the paper's own logistic regression on 30 (0.91184). WDBC
routinely supports 0.95+ from a standard-scaled linear model; a reported 0.912 is below what the
dataset gives up easily.

So the published "quantum beats classical" margin of +5 points rests on a classical comparator that
under-performs the data, and the honest reading of both rows together is:

- this project's QCNN loses to its own classical baseline by ~6 points at the median, and
- the paper's QCNN wins against a classical baseline that is itself ~5 points below what the data
  supports.

Neither row supports a quantum advantage claim, and this project already declines to make one. The
direction is consistent with **Bermejo et al., PRX Quantum 7, 020304 (2026)**, DOI 10.1103/8qt9-72ts,
which this repository already cites: QCNNs benchmarked on "locally-easy" datasets are effectively
classically simulable, and WDBC is such a dataset.

## The shipped artifact is the worst of the five, on accuracy as well

`references/qcnn-entanglement-ablation.md` established this for minority-class F1 (0.622951 against a
five-seed median of 0.870968). It holds for accuracy too: seed 0's **0.7326** is the bottom of
[0.7326, 0.9535], median 0.9070. Any comparison against a published figure that used only the shipped
number would understate this circuit by ~17 accuracy points.

**It is still not retrained on a better seed.** Choosing a seed by its test score is test-set
selection, which the ablation document refuses for the same reason. The spread is reported instead,
and the shipped artifact stays the one whose command is recorded.

## What would reopen this

A published QCNN result on WDBC that states its qubit count, depth, encoding **and** feature subset,
and reports a held-out test split rather than a selected validation split. Then the comparison could
be made on matched terms instead of annotated ones.
