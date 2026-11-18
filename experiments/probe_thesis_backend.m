function probe_thesis_backend()
% Validate convolution against independent conv2 and measure real inference.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'experiments'),fullfile(root,'utilities'));
rng(19,'twister');
for dilation = [1 2 4]
    x = single(rand(11,13,3)); weights = single(randn(3,3,3,4)/10);
    bias = single(randn(4,1));
    actual = thesis_conv(x,weights,bias,dilation,dilation,1);
    reference = zeros(11,13,4);
    for oc = 1:4
        reference(:,:,oc) = double(bias(oc));
        for ic = 1:3
            expanded = zeros(2*dilation+1);
            expanded(1:dilation:end,1:dilation:end) = double(weights(:,:,ic,oc));
            reference(:,:,oc) = reference(:,:,oc)+conv2(double(x(:,:,ic)),rot90(expanded,2),'same');
        end
    end
    maxError = max(abs(double(actual(:))-reference(:)));
    assert(maxError < 2e-5,'Native convolution failed reference comparison.');
    fprintf('CONV_REFERENCE dilation=%d max_error=%g\n',dilation,maxError);
end
image = im2single(imread(fullfile(root,'testsets','BSD68','test001.png')));
input = image+single(25/255)*single(randn(size(image)));
tic; output = thesis_denoise(input,25,'native'); elapsed=toc;
[psnrValue,ssimValue] = Cal_PSNRSSIM(im2uint8(image),im2uint8(output),0,0);
fprintf('REAL_INFERENCE seconds=%.3f PSNR=%.4f SSIM=%.6f\n',elapsed,psnrValue,ssimValue);
fprintf('BACKEND_PROBE_COMPLETE\n');
end
