function probe_matconvnet()
% Verify the compiled library and the supplied grayscale and RGB networks.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'experiments'),fullfile(root,'utilities'));
Setup_thesis_matconvnet(false);
rng(19,'twister');
for dilation = [1 2 4]
    x = single(rand(11,13,3)); w = single(randn(3,3,3,4)/10);
    b = single(randn(4,1));
    actual = vl_nnconv(x,w,b,'pad',dilation,'dilate',dilation);
    reference = zeros(11,13,4);
    for oc = 1:4
        reference(:,:,oc) = double(b(oc));
        for ic = 1:3
            expanded = zeros(2*dilation+1);
            expanded(1:dilation:end,1:dilation:end) = double(w(:,:,ic,oc));
            reference(:,:,oc) = reference(:,:,oc)+ ...
                conv2(double(x(:,:,ic)),rot90(expanded,2),'same');
        end
    end
    err = max(abs(double(actual(:))-reference(:)));
    assert(err < 2e-5,'Compiled convolution differs from conv2.');
    fprintf('COMPILED_CONV dilation=%d max_error=%g\n',dilation,err);
end
for channels = [1 3]
    input = single(rand(32,35,channels));
    actual = thesis_denoise(input,25,'matconvnet');
    reference = thesis_denoise(input,25,'native');
    err = max(abs(double(actual(:))-double(reference(:))));
    assert(err < 2e-4,'Pretrained network outputs disagree.');
    fprintf('PRETRAINED_NETWORK channels=%d max_error=%g\n',channels,err);
end
clean = im2single(imread(fullfile(root,'testsets','BSD68','test001.png')));
input = clean+single(25/255)*single(randn(size(clean)));
tic; output = thesis_denoise(input,25,'matconvnet'); elapsed = toc;
[p,s] = Cal_PSNRSSIM(im2uint8(clean),im2uint8(output),0,0);
fprintf('IRCNN_MATCONVNET CPU seconds=%.3f PSNR=%.4f SSIM=%.6f\n',elapsed,p,s);
fprintf('MATCONVNET_MODEL_CHECK_PASSED\n');
end
