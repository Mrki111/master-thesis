function cfg = thesis_config(profile, device)
% Fixed protocol; parameter choices are evaluated on validation images only.
if nargin < 1, profile = 'milestone'; end
if nargin < 2, device = 'cpu'; end
cfg.profile = lower(profile); cfg.device = lower(device);
cfg.seed = 20261010; cfg.testLimit = Inf; cfg.validationLimit = Inf;
cfg.cropSize = Inf; cfg.exampleCount = 3;
cfg.smokeOnly = strcmp(cfg.profile,'smoke');
template = make_case('denoising',1,'gaussian',25,1);
cfg.cases = template([]);
if any(strcmp(cfg.profile,{'milestone','smoke','denoising','full'}))
    levels = 25;
    channels = 1;
    if any(strcmp(cfg.profile,{'denoising','full'})), levels = [15 25 50]; channels = [1 3]; end
    for ch = channels
        for level = levels
            cfg.cases(end+1) = make_case('denoising',ch,'gaussian',level,1); %#ok<AGROW>
        end
        if ~strcmp(cfg.profile,'milestone')
            for peak = [20 50]
                cfg.cases(end+1) = make_case('denoising',ch,'poisson',peak,1); %#ok<AGROW>
            end
            for probability = [.05 .15]
                cfg.cases(end+1) = make_case('denoising',ch,'impulse',probability,1); %#ok<AGROW>
            end
        end
    end
end
if any(strcmp(cfg.profile,{'deblurring','full','smoke'}))
    for degradation = {'gaussian','motion'}
        cfg.cases(end+1) = make_case('deblurring',1,degradation{1},2.55,1); %#ok<AGROW>
    end
end
if any(strcmp(cfg.profile,{'superresolution','full','smoke'}))
    scales = [2 3 4];
    if cfg.smokeOnly, scales = 2; end
    for scale = scales
        for degradation = {'bicubic','Gaussian'}
            cfg.cases(end+1) = make_case('superresolution',3,degradation{1},0,scale); %#ok<AGROW>
        end
    end
end
assert(~isempty(cfg.cases),'thesis:Profile', ...
    'Profiles: milestone, smoke, denoising, deblurring, superresolution, full.');
assert(any(strcmp(cfg.device,{'cpu','gpu'})),'thesis:Device','Use cpu or gpu.');
if cfg.smokeOnly
    cfg.testLimit = 1; cfg.validationLimit = 1; cfg.cropSize = 64;
    cfg.exampleCount = 1;
end
end

function c = make_case(task,channels,degradation,level,scale)
c.task = task; c.channels = channels; c.degradation = degradation;
c.level = level; c.scale = scale; c.kernel = []; c.iterations = 30;
c.border = 0;
if channels == 1, c.dataset = 'BSD68'; c.validation = 'Set12';
else c.dataset = 'CBSD68'; c.validation = 'Val20'; end
if strcmp(task,'deblurring')
    c.dataset = 'Set3G'; c.validation = 'Val20';
    if strcmp(degradation,'motion'), c.kernel = fspecial('motion',15,30);
    else c.kernel = fspecial('gaussian',25,1.6); end
elseif strcmp(task,'superresolution')
    c.dataset = 'Set5'; c.validation = 'Set14'; c.border = scale;
end
c.id = sprintf('%s_%s_%s_%g_x%d',task,c.dataset,degradation,level,scale);
c.id = strrep(c.id,'.','p');
end
