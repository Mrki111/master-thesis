# Thesis experiment protocol

The mentor-review supplement is documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
It retains the original protocol below for historical reproducibility, adds
BM3D on grayscale Gaussian observations, repeats the RGB baseline scores and
checks an expanded validation grid in a separate output directory. The current
RGB denoising loader requires actual three-channel input; only SR retains the
grayscale-to-three-channel convenience rule.

The completed supplement is `results/thesis/mentor_review_20261118`:
204 BM3D measurements, 476 verified RGB observations, 1,428 reproduced RGB
baseline scores and 160 additional validation candidates. Eleven of 20 reviewed
condition/method selections changed; 488 baseline rows were recomputed and
4,392 retained from the Fourier-corrected run. All 5,084 merged rows passed
artifact verification. The overall counts remain 18/22 conditions for both
IRCNN mean-PSNR and mean-SSIM superiority. BM3D narrows the grayscale Gaussian
PSNR advantage to 0.56–0.58 dB. The thesis presents these supplements separately
from the preserved original comparison tables.

The runner uses the repository's supplied pretrained IRCNN networks through
compiled MatConvNet. It has no training step and no native-backend fallback.
See [MATCONVNET_SETUP.md](MATCONVNET_SETUP.md) for the verified GPU setup.

Run from the repository root in MATLAB:

```matlab
Run_thesis_experiments('milestone','gpu');  % 68 full BSD68 images, Gaussian sigma 25
Run_thesis_experiments('smoke','gpu');      % all tasks on one small image per case
```

The completed initial comparison is in
[`results/thesis/milestone_20261010T163051/REPORT.md`](results/thesis/milestone_20261010T163051/REPORT.md).
Across 68 BSD68 images at Gaussian sigma 25, IRCNN achieved mean PSNR
29.1442 dB and SSIM 0.825044; Wiener 26.3843/0.702053, median
24.8951/0.581166, and bilateral 26.7635/0.683197. The selected bilateral
setting was spatial sigma 1 and range multiplier 3. This run uses the expanded
validation grid.
All 340 per-image method rows and their saved noise realizations passed
artifact verification.

Use `'cpu'` as the second argument to use compiled MatConvNet on CPU.
Every run creates a new directory under `results/thesis/`. An interrupted
run retains its per-image checkpoint with `complete=false` in `state.mat`.
Never report smoke outputs as thesis measurements.

## Scope and profiles

| Profile | Tests | Validation for parameter selection |
|---|---|---|
| milestone | BSD68 grayscale, Gaussian sigma 25 | Set12 |
| denoising | BSD68 grayscale and CBSD68 RGB; Gaussian 15/25/50, Poisson peaks 20/50, impulse probabilities 0.05/0.15 | Set12 / Val20 |
| deblurring | Set3G, Gaussian PSF (25x25, sigma 1.6) and motion PSF (length 15, angle 30 degrees); Gaussian observation noise 2.55/255 | Val20 converted to grayscale |
| superresolution | Set5 RGB, scales 2/3/4; bicubic and Gaussian degradation | Set14 (no free baseline parameters) |
| full | All three task profiles | As above |
| smoke | One image per condition, top-left crops up to 64 pixels | One validation crop per condition |

The full profiles are available but must be run and inspected before their
measurements are used. Smoke verifies execution and metadata; it does not
establish method performance. Running a profile does not write the thesis or
automatically justify its hypotheses.

## Fairness and reproducibility

- Read images as normalized single precision. The historical loader could
  replicate grayscale inputs across RGB channels; the RGB audit confirms that
  all CBSD68 test inputs were actual color files. New RGB denoising runs reject
  grayscale files. SR may still replicate a grayscale input. Crop HR references to
  a multiple of scale, as in the original demos.
- Generate one observation per image/condition with a stable seed derived
  from the fixed base seed, condition, dataset, and image name. Validation and
  test seeds have separate offsets. Every method receives the same observation.
- Save every clean reference, floating-point observation, kernel, and seed in
  `*_observation.mat`. Gaussian/Poisson observations remain unclipped during
  restoration. Salt-and-pepper noise uses a shared spatial RGB mask.
- Compare processed full-image pixel hashes for validation/test duplicates.
  Exact duplicate checking cannot establish independence from the pretrained
  model's original training data or detect every transformed duplicate.
- Select tunable baseline settings by validation mean PSNR only. No test
  reference or test metrics enter `thesis_restore`.
- Denoising: Wiener windows 3/5/7, median windows 3/5/7, bilateral spatial
  sigmas 1/2 and range multipliers 0.75/1.5/3 of nominal observation noise SD.
  Bilateral and median use symmetric boundaries; bilateral is per channel.
  `wiener2` retains MATLAB's standard local-statistics boundary behavior.
- The Gaussian IRCNN model uses the original `ceil(sigma/2)` selection, clipped
  to 1..25. Thus sigma 25 selects model 13. For Poisson/impulse robustness
  experiments, model sigmas 5/15/25/35/50 are selected on validation data.
  There is no retraining or claim that these weights were trained for non-Gaussian noise.
- Deblurring: periodic convolution; true PSF supplied to IRCNN, Wiener, and
  circular Richardson-Lucy. Wiener NSR multipliers 0.25/1/4 and RL iteration
  counts 10/20/40 are selected on validation data.
- Blind deconvolution uses MATLAB `deconvblind`, a fixed 25x25 Gaussian initial
  PSF (sigma 2), and validation-selected iterations 10/20/40. It receives less
  kernel information and has different boundary handling. Its rows explicitly
  say `KernelInformation=estimated`; describe these limitations in the thesis.
  Gaussian observation noise also differs from RL's Poisson likelihood.
- IRCNN deblurring follows `Demo_deblur`'s 30 iterations, denoiser sigma 49 down
  to 13, and lambda equal to observation noise variance/3. The blur/noise
  conditions are declared experimental choices, not claims to reproduce every
  table in the original paper.
- SR follows `Demo_SISR_RGB`'s back-projection: 30 outer iterations, 5 inner
  iterations, alpha 1.75, sigma 12*scale down to scale. The LR observations are
  quantized to uint8, as in that demo. Bicubic and periodic Fourier interpolation
  are baselines. Fourier compensates sampling-grid alignment but does not invert
  Gaussian blur. GPU and CPU SR degradation operators were checked at all scales.
- Apply clipping/uint8 quantization to every output before the same repository
  PSNR/SSIM function. Denoising/deblurring border=0; SR border=scale. RGB PSNR
  uses all RGB pixels; RGB SSIM averages the three channel scores. These are
  not Y-channel measurements.
- Timings include processing and CPU/GPU transfers with device synchronization.
  They exclude model loading, warm-up, image I/O, and metrics. Libraries and fixed
  HQS methods are warmed on validation data. Traditional methods run on CPU;
  IRCNN uses the selected device. Describe this hardware distinction when
  discussing speed. One deterministic noise realization is used per image.

## Outputs

- `per_image.csv`: PSNR, SSIM, seconds, seed, method settings, dimensions,
  metric border, execution device, kernel information, and denoiser model index.
  Model index 0 means no single denoiser index applies (baseline or HQS schedule).
- `summary.csv`: mean and standard deviation across images, mean seconds,
  paired mean IRCNN PSNR gain and number of images where its PSNR is higher.
- `validation.csv` and `parameters.csv`: every candidate validation score and
  the selected setting. Fixed-parameter methods have no tuning score.
- `state.mat`: configuration, completion flag, model and convolution-binary
  SHA-256 hashes, MATLAB/backend information, results, and tuning records.
- `source/`: snapshots of runner, setup, helpers, utilities, and compatibility header.
- `REPORT.md`, condition summary charts, and example reference/observation/method
  PNGs for the first three test images.

Run numerical regressions with `Test_thesis_experiments`. To also verify the
41 task/method results of a completed smoke run:

```matlab
Test_thesis_experiments('results/thesis/<completed-smoke-directory>');
```

`experiments/thesis_conv.m` and its explicit native backend remain diagnostic
tools. They are never used by `Run_thesis_experiments` or its thesis results.

Attribute IRCNN and the supplied weights to Zhang et al., **Learning Deep CNN
Denoiser Prior for Image Restoration**, CVPR 2017, and cite the original
PSNR/SSIM and traditional restoration methods used in your thesis.
