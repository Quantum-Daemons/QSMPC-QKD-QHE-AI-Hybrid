# Addendum to the 3D nodule student's pre-registration: how the scans were fetched

Written on 2026-11-24 while the scans were still being fetched, before any teacher, student or probe was trained and
before any crop was read by a model. The pre-registration (`references/medical-nodule3d-square-student-preregistration.md`)
is unchanged, and `scripts/train/medical_nodule3d_student.py` still refuses to run unless its SHA-256 is the committed
one. Three things are done differently from its text or its plan, for the reasons below; none changes a label, the
split, an arm, a metric or a criterion.

## 1. The series fetched: 883, not 1,018

The pre-registration says all 1,018 CT series are used. A crop exists only where pylidc holds a cluster of readers'
marks, and those clusters, read from pylidc's own database with no image (`scripts/data/lidc_records.py`), lie in 883
series of 875 patients: 2,651 clusters. The other 135 series carry no cluster and give no crop, so fetching them would
add nothing to any arm; they are not fetched. The split was drawn from those records before any image was fetched
(`docs/measurements/medical-nodule3d-split.json`, f7c42bd). Every file of every fetched series is still checked against
the MD5 that TCIA's API lists with it, and each series' file count against TCIA's `ImageCount`
(`scripts/data/lidc_stream.py`); a series that fails either check is not used.

## 2. The order: TotalSegmentator during the fetch

The data drive failed on 2026-11-24 while the collection was being downloaded, and its file system stayed damaged
after repair, so on the owner's choice that day each patient's series are fetched to the system disk, cut into crops,
and deleted (`scripts/data/lidc_stream.py`). TotalSegmentator's size-only candidate needs each training scan once, so it
runs on each training patient's series while they are on disk, in its own environment
(`scripts/train/medical_nodule3d_teacher_totalseg.py worker`), instead of in a pass over a stored collection after the
download. Its input (the DICOM series, read by SimpleITK), its weights (release `v2.5.0-weights`, `Dataset913`, with
the 6 mm `Dataset298` model it crops with) and its output (the volume of the nodule-label component nearest each
nodule's centre) are as pre-registered; only validation and test patients are never segmented, as before. The task
writes two labels, lung (1) and lung nodules (2); the first four volumes were taken from every labelled voxel, which
measured whole lungs, and were recomputed from label 2 alone before any volume was used.

## 3. The training patients' crops, cut a second time

Written on 2026-11-24 after the fetch ended and before any crop was read by a model. The stream wrote each patient's
crops and, for training patients, the 64 mm HU cubes as parts named by `Path.with_suffix`, which made both names the
same file: for the 518 training patients (522 series, 1,719 crops) the HU cubes overwrote the 28 mm crops. Nothing
else was lost: the 932 crops of the 357 validation and test patients, whose records equal the cluster records the
split was drawn from in every field, the HU cubes, the nodules' centres and the TotalSegmentator volumes (522 series,
no error) were each in a file of their own. On the owner's choice that day the training patients' series were fetched
again from TCIA's API and checked as before (each file's MD5 against TCIA's list, each series' file count against
`ImageCount`), and their crops were cut again by the same function (`scripts/data/lidc_stream.py recut`). Three checks
a patient: the re-cut crops' records equal the cluster records, their number equals the first run's, and each crop
equals the centre of the first run's HU cube (the same 28 sample points a side, kept there in float16) within one
uint8 level. Deriving the crops from the HU cubes instead was not done: float16 rounding could move up to 7.3 % of
their voxels by one level. The crops are therefore the ones the pre-registration's procedure makes; the stream's
summary (`docs/measurements/lidc-stream-2026-11-24.json`) records both fetches. The outcome, the same day and before any
model was trained: all 883 series (215,912 files) verified in the first fetch and the training patients' 522 series
(127,372 files) again in the second; 1,719 crops re-cut, every patient's records equal to the cluster records, every
crop within one uint8 level of its HU cube's centre (2.3 % of the voxels one level off, the float16 rounding); and the
merged crops give the split's counts in every part (train 1,178 labelled nodules, 381 malignant; validation 170, 55;
test 347, 109).

## 4. The crops the page shows

Written on 2026-11-24 before any teacher, student or probe was trained. If the candidate is adopted (criteria 1 and
2), the medical page ships twelve test crops for its card, chosen by this rule and no other: the labelled test crops
in the order of the cluster records (patient, series, cluster), the first six with a malignant label and the first six
with a benign one, the rule the 2D square student's card uses for its eight images. No patient or series identifier is
shipped with them, only each crop's label, its readers' count and their median malignancy rating; the page's figures
for the whole test part are copied from the record, not computed on these twelve.
