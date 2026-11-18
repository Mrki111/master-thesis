# Local MATLAB and MatConvNet setup

Verified on this machine: MATLAB R2015a, Visual Studio 2015, CUDA 11.0
(`nvcc` 11.0.194), cuDNN 8.0.1, and NVIDIA RTX 4070 Laptop GPU (8 GB).
The installed CUDA root is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0`.
The cuDNN root is its nested `cuda` directory, with `include`, `lib/x64`, and `bin`.

MatConvNet is an external dependency and is not stored in this repository.
Obtain the documented source and reviewed binaries from the companion archive
and place its CPU/GPU build directories under `matconvnet/`, or supply the source
and compile it as described below. Dependency directories are ignored by Git.

From this repository's root in MATLAB:

```matlab
Setup_thesis_matconvnet(false, [], 'gpu');
addpath('experiments', 'utilities');
probe_matconvnet_gpu;
```

To select the preserved CPU build:

```matlab
Setup_thesis_matconvnet(false);
```

For the experiment helper, GPU input selects GPU execution:

```matlab
y = gpuArray(im2single(imread('testsets/BSD68/test001.png')));
z = gather(thesis_denoise(y, 25, 'matconvnet'));
```

The example above passes a clean image only to show the API; noise must be
added for an actual denoising comparison. The existing pretrained weights
are used without training.

## Rebuilding

```matlab
Setup_thesis_matconvnet(true);             % CPU
Setup_thesis_matconvnet(true, [], 'gpu');  % GPU with cuDNN
```

Optional arguments 4 and 5 specify different CUDA and cuDNN roots. Argument 6
can be `true` to resume an interrupted build **only when source, compiler flags,
and compatibility headers are unchanged**. Ordinary GPU rebuilding recompiles
all objects to avoid using stale objects after compatibility changes.

The CPU and GPU builds are in separate directories under `matconvnet/`.
The setup does not write into the MATLAB or CUDA installations. Compiling
selects the C++ MEX configuration in MATLAB's user preferences. Old MATLAB's
VS2013 compiler XML is adapted locally for VS2015 in `experiments/setup/`.

## Changes from the README's historical GPU recipe

The README documents CUDA 8.0 and cuDNN 5.1. This installation uses CUDA 11.0
and cuDNN 8.0.1. `experiments/setup/cudnn8_compat.hpp` adapts the removed cuDNN
algorithm selection APIs to the v7 heuristic APIs, respecting workspace limits.
It is included at compile time; the MatConvNet source and model files are
unchanged. CUDA kernels include compute-7.5 PTX for execution on the newer GPU.
The setup also adapts CUDA optimization flags and R2015a library path handling.

NVIDIA documents [the removed cuDNN APIs](https://archive.docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/api/index.html#cudnn-api-changes)
and [PTX compatibility with Ada GPUs](https://docs.nvidia.com/cuda/archive/13.0.3/ada-compatibility-guide/index.html).

## Verification

The full experiment's saved `state.mat` independently records MATLAB
`8.5.0.197613 (R2015a)`, the GPU name and compute capability, the external CUDA
and cuDNN paths and SHA-256 of the loaded convolution MEX binary. The binary
hash is `7859c1438dd555e56a1d2e9206e7c70d4fcd535407b416c2f6d19e0715cba58d`.
MATLAB's `gpuDevice.ToolkitVersion` value of **6.5** describes MATLAB's bundled
toolkit, while **CUDA 11.0** describes the toolkit used for the external MEX
build. These are separate version fields. This installation was locally
adapted and checked numerically; it is not a claim of official support for
this software/hardware combination. Reproduction and committee handover
instructions are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

- `logs/thesis_cuda_check.log` (in the companion archive): standalone CUDA kernel, cuBLAS multiplication, and
  cuDNN convolution passed on the RTX 4070.
- `logs/thesis_matconvnet_gpu_link.log` (in the companion archive): GPU MEX linking and startup check passed.
- `logs/thesis_matconvnet_gpu_model_check.log` (in the companion archive): dilations 1/2/3/4 checked on GPU,
  with and without cuDNN; grayscale and RGB supplied networks checked at
  denoiser sigma 2/25/50 against CPU inference.
- Full-image sigma-25 check: CPU PSNR 25.91000 dB, GPU 25.90999 dB;
  CPU SSIM 0.7274436, GPU 0.7274428. Maximum error in the small network tests
  was below `8e-7` in normalized pixel units.
- One warmed full-image inference: CPU 0.387233 s, GPU 0.022637 s.
  GPU timing synchronizes the device and excludes transfers. This is a setup
  check on one image, not a benchmark average or thesis result.

IRCNN's dilated layers may use MatConvNet's CUDA/cuBLAS implementation even
when cuDNN is enabled, because beta25's cuDNN path does not implement dilation.
The GPU tests cover both convolution paths. The CPU build remains available.
