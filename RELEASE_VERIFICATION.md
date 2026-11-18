# Code and configuration handover verification 18 November 2026

Use the version 2 archives in `release/`. The first handovers are retained as
historical copies. No thesis measurements, model weights, degradation parameters,
restoration computations or metric definitions were changed by this check.

The original archive checksums and every manifest entry passed verification.
The full archive was then extracted to a separate directory and executed there
with MATLAB R2015a and the supplied CPU MatConvNet build.

| Check | Result |
| --- | --- |
| Fresh CPU smoke experiment | All 41 task/method rows passed |
| Comparison with the recorded CPU smoke experiment | Maximum absolute PSNR and SSIM differences both 0 |
| Original full-run artifact and numerical checks | Passed |
| Supplemental review verification in the relocated copy | All 5,084 rows passed |
| Configurations | 22 conditions; 16 expanded bilateral candidates |
| Windows and Unix legacy parent-directory references | Relocation regression checks passed |
| Preservation during verification | All 1,864 result/model files in the extracted test copy unchanged |

One portability defect was found: verification used an absolute parent-run path
saved on the author's machine. `experiments/thesis_parent_run.m` now prefers the
corresponding run under the extracted package's `results/thesis/` directory.
Historical states retain their original provenance. The verification command
also leaves previously verified source snapshots and records unchanged.

The regression tests exercised Windows and Unix path strings on the documented
Windows installation. This does not establish execution on Linux or on another
MATLAB/CUDA version. The fresh execution check covered CPU smoke reproduction
and saved-artifact verification; it did not rerun the entire GPU experiment or
all BM3D measurements. Their earlier evidence remains supplied in the archive.

`IRCNN_kod_i_konfiguracije_20261118_v2.zip` is the compact code-review package.
It omits pretrained weights, datasets, saved observations and compiled binaries.
`IRCNN_reprodukcija_20261118_v2.zip` supplies those research inputs and the reviewed
Windows MEX binaries. Use the full archive for execution. MATLAB, the documented
NVIDIA dependencies when using GPU, and the restricted BM3D implementation are
obtained separately, as explained in `REPRODUCIBILITY.md`.

The execution log is `logs/package_relocation_check.log`. Run
`tools/verify_release.py` with Python 3.11 or newer to check each archive's SHA-256,
manifest coverage and all member hashes. Packaging tools support explicit output
paths and refuse to overwrite a preserved archive.
