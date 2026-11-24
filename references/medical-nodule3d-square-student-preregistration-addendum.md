# Addendum to the 3D nodule student's pre-registration: how the scans were fetched

Written on 2026-11-24 while the scans were still being fetched, before any teacher, student or probe was trained and
before any crop was read by a model. The pre-registration (`references/medical-nodule3d-square-student-preregistration.md`)
is unchanged, and `scripts/train/medical_nodule3d_student.py` still refuses to run unless its SHA-256 is the committed
one. Two things are done differently from its text, for the reasons below; neither changes a label, the split, an arm,
a metric or a criterion.

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
