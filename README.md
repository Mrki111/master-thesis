# Master thesis experiment code and configurations

MATLAB experiment code accompanying Lazar Mrkic's thesis on CNN image denoising
and restoration. The experiments use pretrained IRCNN models without additional training.

## Included

- Six experiment, setup and verification entry points in the repository root.
- `experiments/`: fixed configurations, restoration code and local setup adapters.
- `utilities/`: the eight helpers required by those experiments and tests.
- Reproduction instructions and source attribution.

MatConvNet, upstream demo projects, models, datasets, saved results, compiled
binaries and the thesis document are supplied separately. Obtain the full
`IRCNN_reprodukcija_20261118_v2.zip` companion archive from the author for the
documented research inputs, MatConvNet dependency and historical verification data.
Place its `models/`, `testsets/`, `kernels/`, `results/` and `matconvnet/`
directories beside these entry points when reproducing the saved experiments.
These dependency and data directories are ignored by Git.

## Running

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md),
[THESIS_EXPERIMENTS.md](THESIS_EXPERIMENTS.md) and
[MATCONVNET_SETUP.md](MATCONVNET_SETUP.md). MATLAB and its required toolboxes are
obtained separately. The BM3D download helper obtains the restricted implementation
separately and checks its reviewed archive hash.

With the documented inputs and compiled dependencies available:

```matlab
Setup_thesis_matconvnet(false, [], 'cpu');
Test_thesis_experiments;
folder = Run_thesis_experiments('smoke', 'cpu');
Test_thesis_experiments(folder);
```

## Attribution

Based on [IRCNN by Kai Zhang, Wangmeng Zuo, Shuhang Gu and Lei Zhang](https://github.com/cszn/IRCNN),
*Learning Deep CNN Denoiser Prior for Image Restoration*, CVPR 2017.
Original notices remain in the helper source. See [THIRD_PARTY.md](THIRD_PARTY.md).
