# Reproducing the thesis experiments

Run commands from the repository root. The MATLAB experiment code uses the
supplied pretrained IRCNN weights without training. The original authors are
Zhang, Zuo, Gu and Zhang, *Learning Deep CNN Denoiser Prior for Image Restoration*,
CVPR 2017. Preserve the attribution and third-party licence notices.

## Software and hardware

The actual saved full-run state reports MATLAB **8.5.0.197613 (R2015a)**,
MatConvNet **1.0-beta25**, NVIDIA **GeForce RTX 4070 Laptop GPU**, compute
capability **8.9**, and the external CUDA **11.0** installation. The build probe
reports nvcc **11.0.194** and cuDNN runtime **8001 (8.0.1)**. This is a locally
adapted build, not an unmodified or officially supported installation recipe.
See [MATCONVNET_SETUP.md](MATCONVNET_SETUP.md) for the cuDNN adapter, compiler
configuration and compute-7.5 PTX build flags. MATLAB's `gpuDevice.ToolkitVersion`
reports its own bundled toolkit (**6.5**); it is not the CUDA toolkit used to
compile the external MatConvNet MEX binary. The saved MATLAB driver API value
is also distinct from the installed NVIDIA display-driver release number.

Required MATLAB toolboxes are Image Processing, Statistics and Machine Learning
(`poissrnd`) and, for GPU execution, Parallel Computing. MATLAB, CUDA, cuDNN and
Visual Studio must be obtained separately. CPU MatConvNet execution is available
through the `cpu` argument; CPU timings will differ from the recorded GPU run.

## Original experiment and correction

```matlab
Setup_thesis_matconvnet(false, [], 'gpu');
Test_thesis_experiments;
outputFolder = Run_thesis_experiments('full','gpu');
Test_thesis_experiments(outputFolder);
```

The historical full run is `results/thesis/full_20261011T180756`. Its exact source
snapshot is in `source/`; model, binary and environment fingerprints are in
`state.mat`. Fixed seeds, validation/test data, degradation operators, quantization,
metric borders and timing scope are documented in [THESIS_EXPERIMENTS.md](THESIS_EXPERIMENTS.md).
Use the archived source when reproducing that historical version verbatim.
Apply that snapshot to a separate copy of the repository, retaining the models,
datasets and utilities at the repository root; the `source/` directory alone
is not a standalone experiment directory.

The corrected Fourier implementation is already in the current source. To
reproduce the historical correction on the saved observations:

```matlab
Reevaluate_thesis_fourier('results/thesis/full_20261011T180756', ...
    'results/thesis/fourier_alignment_reproduced');
```

## Mentor review

Download the authors' **legacy BM3D v2.01**, rather than silently substituting a
different implementation. The download helper pins the SHA-256 of the reviewed
archive. The [authors' website](https://webpages.tuni.fi/foi/GCF-BM3D/) and
[licence](https://webpages.tuni.fi/foi/GCF-BM3D/legal_notice.html) remain authoritative.
The package is for non-commercial research and is excluded from the distributable
archive and version control because network redistribution requires permission.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Install_thesis_bm3d.ps1
```

The review reads the unchanged historical full run and Fourier correction. Run
the stages in this order, using a new output folder for an independent repetition:

```matlab
folder = 'results/thesis/mentor_review_reproduced';
Review_thesis_experiments('bm3d',folder);
Review_thesis_experiments('audit',folder);
Review_thesis_experiments('tuning',folder);
Test_thesis_experiments(folder);
Verify_thesis_review(folder);
```

Saved states preserve their original provenance paths. Current verification
resolves the parent run under the extracted package's `results/thesis/` directory
before considering its old location. This allows the complete archive to be
extracted to a different folder without editing historical states. Verifying
an already verified review does not overwrite its source snapshot or records.

BM3D receives only the saved noisy grayscale image, known Gaussian sigma and
fixed normal profile: `BM3D(1,double(y),sigma,'np',0)`. The scalar 1 withholds the
clean image. All 68 BSD68 images at each sigma 15/25/50 are evaluated with the
same uint8 output/metric convention. RGB BM3D is outside this supplemental test.
The supplied implementation automatically uses its strong-noise parameter
settings when sigma exceeds 40, including the normal-profile call at sigma 50.

The RGB audit checks all 476 saved CBSD68 observations against the actual color
files and their regenerated noise. It repeats all 1,428 Wiener/median/bilateral
RGB results and checks the single-channel path, combined RGB MSE and mean channel
SSIM. RGB denoising now rejects a single-channel source instead of silently
replicating it. The historical SR grayscale-to-three-channel rule is preserved.

The expanded validation grid retains the original candidates, adds bilateral
spatial sigma 3 and range multipliers 6/12, and includes the infinite-range
Gaussian spatial-filter limit. Wiener-deconvolution multipliers extend through
16/64/256/1024; Richardson–Lucy iterations through 80/160/320; blind-deconvolution
iterations include 1/2/5 below the original lower bound. Only validation PSNR
selects parameters. Test outputs are then recomputed only where the selection
changes. A boundary selection still does not establish a global optimum.

`bm3d_summary.csv`, `rgb_inputs.csv`, `rgb_reproduced.csv`, `grid_changes.csv`,
`validation.csv`, `parameters.csv` and `expanded_summary.csv` retain the evidence.
The merged `state.mat` records inherited rows separately through its parent-run
reference. Timings combine historical unchanged rows and newly measured changed
rows; they are not a fresh controlled benchmark. The original run is never edited.

For this first review execution, configuration labels added while MATLAB had
the runner cached were attached after completion with
`experiments/annotate_review_configuration.m`. The raw checkpoint remains in
`state_before_config_annotation.mat`; all 5,084 measurements and tuning scores
were preserved exactly. Fresh runs with the supplied final source write these
labels and the fixed BM3D selections directly.

## Companion archive

This Git repository contains the thesis experiment code and configurations only.
Obtain `IRCNN_reprodukcija_20261118_v2.zip` from the author for MatConvNet, original
models, datasets, kernels, saved observations, CSV/MAT records and verification logs.
Place its dependency and data directories under the repository root as described
in the README, retaining the required licence notices. MATLAB, CUDA/cuDNN where
applicable, and the restricted BM3D package are obtained separately.

The full archive also contains historical source snapshots, the one-time
configuration-annotation script and optional packaging/checksum tools. Those
supporting artifacts are not part of this minimal code/configuration repository.
