# Source attribution and external dependencies

The experiments build on IRCNN by Kai Zhang, Wangmeng Zuo, Shuhang Gu and
Lei Zhang, CVPR 2017: https://github.com/cszn/IRCNN.

Required helper files retain their original author and permission notices,
including the PSNR/SSIM implementation, sampling operators and Shepard interpolation.
Those notices apply to the corresponding source and are not replaced here.

MatConvNet 1.0-beta25 is an external dependency. Its source, demos and compiled
binaries are not included in this repository. Preserve its `COPYING` notice when
obtaining it from its authors or the companion archive. The local cuDNN adapter
and setup configuration are included under `experiments/setup/`.

BM3D, pretrained weights, datasets, MATLAB and NVIDIA libraries are obtained
separately under their respective terms. This repository grants no new licence
to third-party material.
