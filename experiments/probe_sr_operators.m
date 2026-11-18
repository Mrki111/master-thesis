function probe_sr_operators()
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'utilities'),fullfile(root,'experiments'));
Setup_thesis_matconvnet(false,[],'gpu');
rng(7,'twister'); input = single(rand(48,60,3));
for scale = [2 3 4]
    for type = {'bicubic','Gaussian'}
        cpu = imresize_down(input,scale,type{1},1.6);
        gpu = gather(imresize_down(gpuArray(input),scale,type{1},1.6));
        error = max(abs(cpu(:)-gpu(:)));
        fprintf('SR_OPERATOR type=%s scale=%d max_error=%g\n',type{1},scale,error);
        assert(isequal(size(cpu),size(gpu)) && error < 2e-5, ...
            'GPU forward operator disagrees with CPU degradation.');
    end
end
fprintf('SR_OPERATOR_CHECK_PASSED\n');
end
