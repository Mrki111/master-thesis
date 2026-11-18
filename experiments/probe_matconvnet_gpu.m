function probe_matconvnet_gpu()
% Check actual GPU MEX inference against CPU inference of the same weights.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'experiments'),fullfile(root,'utilities'));
info = Setup_thesis_matconvnet(false,[],'gpu');
device = gpuDevice();
fprintf('GPU_MODEL_CHECK device=%s binary=%s\n',device.Name,info.convolutionBinary);
rng(97,'twister');
for dilation = [1 2 3 4]
    x = single(randn(19,23,3)); w = single(randn(3,3,3,4)/10); b = single(randn(4,1));
    reference = vl_nnconv(x,w,b,'pad',dilation,'dilate',dilation);
    for mode = {'CuDNN','NoCuDNN'}
        actual = gather(vl_nnconv(gpuArray(x),gpuArray(w),gpuArray(b), ...
            'pad',dilation,'dilate',dilation,mode{1}));
        err = max(abs(double(actual(:))-double(reference(:))));
        assert(err < 2e-4,'GPU convolution differs from CPU.');
        fprintf('GPU_CONV %s dilation=%d max_error=%g\n',mode{1},dilation,err);
    end
end
for channels = [1 3]
    x = single(rand(40,43,channels));
    for sigma = [2 25 50]
        reference = thesis_denoise(x,sigma,'matconvnet');
        actual = gather(thesis_denoise(gpuArray(x),sigma,'matconvnet'));
        err = max(abs(double(actual(:))-double(reference(:))));
        assert(err < 5e-4,'GPU IRCNN differs materially from CPU IRCNN.');
        fprintf('GPU_PRETRAINED channels=%d sigma=%d max_error=%g\n',channels,sigma,err);
    end
end
clean = im2single(imread(fullfile(root,'testsets','BSD68','test001.png')));
input = clean+single(25/255)*single(randn(size(clean)));
% Warm both paths before measuring inference and synchronize GPU work.
reference = thesis_denoise(input,25,'matconvnet');
gpuInput = gpuArray(input);
thesis_denoise(gpuInput,25,'matconvnet'); wait(device);
tic; reference = thesis_denoise(input,25,'matconvnet'); cpuSeconds = toc;
wait(device); tic; gpuOutput = thesis_denoise(gpuInput,25,'matconvnet');
wait(device); gpuSeconds = toc; actual = gather(gpuOutput);
[cpuPSNR,cpuSSIM] = Cal_PSNRSSIM(im2uint8(clean),im2uint8(reference),0,0);
[gpuPSNR,gpuSSIM] = Cal_PSNRSSIM(im2uint8(clean),im2uint8(actual),0,0);
assert(abs(cpuPSNR-gpuPSNR) < 0.01 && abs(cpuSSIM-gpuSSIM) < 1e-4, ...
    'GPU and CPU denoising metrics differ materially.');
fprintf('CPU_REAL_IMAGE seconds=%.6f PSNR=%.5f SSIM=%.7f\n',cpuSeconds,cpuPSNR,cpuSSIM);
fprintf('GPU_REAL_IMAGE seconds=%.6f PSNR=%.5f SSIM=%.7f\n',gpuSeconds,gpuPSNR,gpuSSIM);
fprintf('GPU_MATCONVNET_MODEL_CHECK_PASSED\n');
end
