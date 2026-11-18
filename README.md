# Master thesis code and configurations

MATLAB code and experiment configurations accompanying Lazar Mrkic's master
thesis on convolutional neural networks for image denoising and restoration.
The experiments use published pretrained IRCNN networks without additional training.

## Contents

- Experiment runners for denoising, deblurring and super-resolution.
- Original and expanded validation grids, fixed seeds and metric conventions.
- BM3D comparison and RGB verification code.
- MatConvNet source, CPU/GPU setup code and the local cuDNN compatibility adapter.
- Numerical tests, saved-result verification and packaging/checksum utilities.

This repository contains source code, configurations and documentation. Models,
datasets, saved observations, experiment records, compiled binaries and the thesis
document are provided separately by the author. The full companion archive is
`IRCNN_reprodukcija_20261118_v2.zip`; its evidence and required research inputs
are needed to reproduce or verify the saved thesis experiments.

## Reproduction

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete protocol and commands,
[THESIS_EXPERIMENTS.md](THESIS_EXPERIMENTS.md) for configurations and metrics, and
[MATCONVNET_SETUP.md](MATCONVNET_SETUP.md) for the documented software environment.
Supply the companion archive's input directories and appropriate MEX binaries,
or compile MatConvNet using the setup instructions. MATLAB and its required
toolboxes are obtained separately. Obtain the restricted legacy BM3D package
through `Install_thesis_bm3d.ps1`; BM3D is not redistributed here.

With those dependencies and inputs available, run from the repository root:

```matlab
Setup_thesis_matconvnet(false, [], 'cpu');
Test_thesis_experiments;
folder = Run_thesis_experiments('smoke', 'cpu');
Test_thesis_experiments(folder);
```

The version 2 handover passed an extracted-copy CPU smoke run (41 task/method
results matching the recorded PSNR/SSIM values) and verification of 5,084 saved
results. See [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) for scope and limits.
Verification resolves parent results relative to a relocated package and preserves
already verified historical records.

## Attribution

Based on [IRCNN by Kai Zhang, Wangmeng Zuo, Shuhang Gu and Lei Zhang](https://github.com/cszn/IRCNN),
*Learning Deep CNN Denoiser Prior for Image Restoration*, CVPR 2017.
The original project README is retained in [IRCNN_README.md](IRCNN_README.md).
MatConvNet's licence notices are retained in its `COPYING` files.
See [THIRD_PARTY.md](THIRD_PARTY.md). No new licence is imposed on third-party code.
