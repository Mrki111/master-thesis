function Test_thesis_experiments(smokeFolder)
% Numeric regressions and optional end-to-end artifact verification.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'experiments'),fullfile(root,'utilities'));
previous = rng; cleanup = onCleanup(@() rng(previous)); %#ok<NASGU>
rng(314,'twister');
for shape = {[3 3],[3 5],[4 6]}
    h = rand(shape{1}); h = h/sum(h(:));
    [G,Gt] = defGGt(h,2);
    x = randn(24,28,3); y = randn(12,14,3);
    gx = G(x); gty = Gt(y);
    left = sum(gx(:).*y(:)); right = sum(x(:).*gty(:));
    fprintf('ADJOINT kernel=%dx%d error=%g\n',shape{1},abs(left-right));
    assert(abs(left-right) < 1e-10*max(1,abs(left)),'G/Gt adjoint regression.');
    eigenvalues = constructGGt(h,2,24,28);
    actual = G(Gt(y)); reference = real(ifft2(bsxfun(@times,fft2(y),eigenvalues)));
    assert(max(abs(actual(:)-reference(:))) < 1e-10,'G*Gt FFT regression.');
end
mask = true(7,9); mask(4,5) = false;
image = cat(3,ones(7,9),2*ones(7,9),3*ones(7,9));
output = shepard_initialize(image,mask);
assert(max(abs(squeeze(output(4,5,:))-[1;2;3])) < 1e-12,'RGB interpolation leaked between channels.');
[~,smallSSIM] = Cal_PSNRSSIM(zeros(5,5,'uint8'),ones(5,5,'uint8'),0,0);
assert(isinf(smallSSIM) && smallSSIM < 0,'Small SSIM must return its documented sentinel.');
% A single perturbed RGB channel must contribute one third of the full MSE.
rgbReference=zeros(16,16,3,'uint8'); rgbOutput=rgbReference; rgbOutput(:,:,1)=3;
[rgbPSNR,rgbSSIM]=Cal_PSNRSSIM(rgbReference,rgbOutput,0,0);
assert(abs(rgbPSNR-10*log10(255^2/3))<1e-12,'RGB PSNR omitted or collapsed channels.');
rgbChannelSSIM=zeros(1,3);
for ch=1:3
    [~,rgbChannelSSIM(ch)]=Cal_PSNRSSIM(rgbReference(:,:,ch),rgbOutput(:,:,ch),0,0);
end
assert(abs(rgbSSIM-mean(rgbChannelSSIM))<1e-12,'RGB SSIM channel mean changed.');
rgbCfg=thesis_config('denoising','cpu');
rgbCase=rgbCfg.cases(find([rgbCfg.cases.channels]==3,1));
try
    thesis_read_clean(fullfile(root,'testsets','BSD68','test001.png'),rgbCase,Inf);
    assert(false,'thesis:MissingRGBCheck','Grayscale input was accepted for RGB denoising.');
catch problem
    assert(strcmp(problem.identifier,'thesis:RGBInput'),'RGB loader did not reject a grayscale file.');
end
srInputCase=rgbCase; srInputCase.task='superresolution'; srInputCase.scale=2;
srReference=thesis_read_clean(fullfile(root,'testsets','BSD68','test001.png'),srInputCase,Inf);
assert(size(srReference,3)==3 && isequal(srReference(:,:,1),srReference(:,:,3)), ...
    'SR grayscale replication contract changed.');
cfg = thesis_config('smoke','cpu'); c = cfg.cases(1);
clean = repmat(single(.3),[128 128]);
state = rng;
[first,meta] = thesis_degrade(clean,c,123);
assert(isequal(state,rng),'Degradation changed caller RNG state.');
second = thesis_degrade(clean,c,123);
assert(isequal(first,second),'Observation generation is not repeatable.');
assert(any(first(:) < 0) || any(first(:) > 1),'Gaussian observation was unexpectedly clipped.');
c.degradation = 'poisson'; c.level = 20;
noisy = thesis_degrade(clean,c,456);
assert(abs(mean(double(noisy(:)))-.3) < .004,'Poisson mean regression.');
assert(abs(var(double(noisy(:)))-.3/20) < .002,'Poisson variance regression.');
c.degradation = 'impulse'; c.level = .15;
rgb = cat(3,clean,clean+single(.1),clean+single(.2));
noisy = thesis_degrade(rgb,c,789);
changed = noisy(:,:,1) ~= rgb(:,:,1);
for channel = 2:3
    assert(isequal(changed,noisy(:,:,channel) ~= rgb(:,:,channel)),'Impulse masks differ by channel.');
end
for level = [2 3 4]
    srCfg = thesis_config('superresolution','cpu'); c = srCfg.cases(1); c.scale = level;
    output = thesis_restore(single(ones(12,14,3)*.4),c,meta,'Fourier',struct(),'cpu');
    assert(isequal(size(output),[12*level 14*level 3]),'Fourier dimensions changed.');
    assert(max(abs(double(output(:))-.4)) < 1e-6,'Fourier does not preserve constants.');
    % Derive the physical sampling phase from the actual degradation operator,
    % then reconstruct a band-limited signal on that grid. Constants alone
    % cannot detect a misplaced interpolation grid.
    side = 96; [rr,cc] = ndgrid(0:side-1,0:side-1);
    expected = .5+.2*cos(4*pi*cc/side)+.1*sin(4*pi*rr/side);
    for degradation = {'bicubic','Gaussian'}
        c.degradation = degradation{1};
        sampledRamp = imresize_down(repmat(single(1:side),side,1),level,c.degradation,1.6);
        phase = double(sampledRamp(5,5))-(4*level+1);
        [lr,lc] = ndgrid(phase+level*(0:side/level-1));
        input = single(.5+.2*cos(4*pi*lc/side)+.1*sin(4*pi*lr/side));
        aligned = thesis_restore(input,c,meta,'Fourier',struct(),'cpu');
        error = max(abs(double(aligned(:))-expected(:)));
        assert(error < 1e-5,'Fourier sampling phase does not match the degradation grid.');
        fprintf('FOURIER_GRID type=%s scale=%d phase=%.6f error=%g\n',c.degradation,level,phase,error);
    end
end
fprintf('THESIS_NUMERIC_TESTS_PASSED\n');
if nargin > 0
    data = load(fullfile(smokeFolder,'state.mat'));
    assert(data.complete,'Experiment run was incomplete.');
    if data.cfg.smokeOnly
        assert(numel(data.rows) == 41,'Smoke output does not cover all task/method pairs.');
    elseif strcmp(data.cfg.profile,'milestone')
        assert(numel(data.rows) == 340,'Milestone must cover 68 images and 5 methods.');
        assert(numel(unique({data.rows.Image})) == 68,'Milestone does not cover BSD68.');
    end
    assert(all(isfinite([data.rows.PSNR])) && all(isfinite([data.rows.SSIM])));
    for k = 1:numel(data.cfg.cases)
        c = data.cfg.cases(k); selected = data.rows(strcmp({data.rows.Case},c.id));
        assert(all([selected.Border] == c.border),'Methods received different metric borders.');
        expected = thesis_methods(c);
        if isfield(data.cfg,'bm3dGrayscaleGaussian') && data.cfg.bm3dGrayscaleGaussian ...
                && strcmp(c.task,'denoising') && c.channels==1 && strcmp(c.degradation,'gaussian')
            expected{end+1}='BM3D';
        end
        names = unique({selected.Image},'stable');
        assert(numel(selected) == numel(expected)*numel(names),'Missing method result.');
        for j = 1:numel(names)
            paired = selected(strcmp({selected.Image},names{j}));
            assert(numel(unique([paired.Seed])) == 1,'Methods received different observation seeds.');
            [~,base] = fileparts(names{j});
            observationPath = fullfile(smokeFolder,c.id,[base '_observation.mat']);
            if exist(observationPath,'file') ~= 2 && isfield(data,'parentRun')
                parentFolder = thesis_parent_run(data,root);
                observationPath = fullfile(parentFolder,c.id,[base '_observation.mat']);
            end
            observation = load(observationPath);
            repeated = thesis_degrade(observation.clean,c,paired(1).Seed);
            assert(isequal(repeated,observation.y),'Saved observation does not match its seed.');
        end
    end
    for k = 1:numel(data.selections)
        s = data.selections(k);
        candidates = data.tuning(strcmp({data.tuning.Case},s.Case) & strcmp({data.tuning.Method},s.Method));
        if isempty(candidates), continue; end
        [~,best] = max([candidates.MeanPSNR]);
        assert(strcmp(s.Parameters,candidates(best).Parameters),'Parameter selection ignored validation scores.');
    end
    fprintf('THESIS_ARTIFACT_TESTS_PASSED profile=%s rows=%d\n',data.cfg.profile,numel(data.rows));
end
end
