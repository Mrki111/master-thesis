function outputFolder = Run_thesis_experiments(profile, device)
%RUN_THESIS_EXPERIMENTS Reproducible comparison of supplied IRCNN and baselines.
% Run_thesis_experiments('milestone','gpu') runs BSD68 Gaussian sigma=25.
% Run_thesis_experiments('smoke','gpu') checks all tasks on small crops.
% Other profiles: denoising, deblurring, superresolution, full.
if nargin < 1, profile = 'milestone'; end
if nargin < 2, device = 'cpu'; end
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'experiments'),fullfile(root,'utilities'));
cfg = thesis_config(profile,device);
backend = Setup_thesis_matconvnet(false,[],cfg.device);
previousRNG = rng; cleanup = onCleanup(@() rng(previousRNG)); %#ok<NASGU>
outputFolder = fullfile(root,'results','thesis', ...
    [cfg.profile '_' datestr(now,'yyyymmddTHHMMSS')]);
suffix = 0; baseFolder = outputFolder;
while exist(outputFolder,'dir')
    suffix = suffix+1; outputFolder = sprintf('%s_%d',baseFolder,suffix);
end
mkdir(outputFolder);
rows = struct([]); tuning = struct([]); selections = struct([]); complete = false;
environment.matlabVersion = version;
environment.backend = backend;
environment.modelgraySHA256 = file_hash(fullfile(root,'models','modelgray.mat'));
environment.modelcolorSHA256 = file_hash(fullfile(root,'models','modelcolor.mat'));
environment.convolutionSHA256 = file_hash(backend.convolutionBinary);
environment.started = datestr(now,30);
snapshot_source(root,outputFolder);
save(fullfile(outputFolder,'state.mat'),'cfg','environment','rows','tuning','selections','complete');
fprintf('THESIS_RUN profile=%s device=%s output=%s\n',cfg.profile,cfg.device,outputFolder);
if cfg.smokeOnly, fprintf('SMOKE ONLY: small crops; do not report these as thesis results.\n'); end

for caseIndex = 1:numel(cfg.cases)
    c = cfg.cases(caseIndex);
    testFiles = image_files(root,c.dataset);
    validationFiles = image_files(root,c.validation);
    check_disjoint(testFiles,validationFiles,c);
    testFiles = testFiles(1:min(numel(testFiles),cfg.testLimit));
    validationFiles = validationFiles(1:min(numel(validationFiles),cfg.validationLimit));
    [methods,grids] = thesis_methods(c);
    fprintf('CASE %s test=%d validation=%d\n',c.id,numel(testFiles),numel(validationFiles));
    warm_models(c,cfg.device);
    valClean = cell(size(validationFiles)); valY = cell(size(validationFiles)); valMeta = cell(size(validationFiles));
    for j = 1:numel(validationFiles)
        valClean{j} = read_clean(validationFiles{j},c,cfg.cropSize);
        seed = image_seed(cfg.seed,c.id,c.validation,validationFiles{j},true);
        [valY{j},valMeta{j}] = thesis_degrade(valClean{j},c,seed);
    end
    selected = cell(size(methods));
    for m = 1:numel(methods)
        scores = zeros(1,numel(grids{m}));
        if numel(grids{m}) > 1
            for p = 1:numel(grids{m})
                imageScores = zeros(1,numel(validationFiles));
                for j = 1:numel(validationFiles)
                    output = thesis_restore(valY{j},c,valMeta{j},methods{m},grids{m}{p},cfg.device);
                    imageScores(j) = metrics(valClean{j},output,c.border);
                end
                scores(p) = mean(imageScores);
                item.Case = c.id; item.Method = methods{m};
                item.Parameters = parameter_key(grids{m}{p});
                item.ValidationDataset = c.validation; item.N = numel(validationFiles);
                item.MeanPSNR = scores(p);
                tuning = append_struct(tuning,item);
                fprintf('TUNE %s %s %s PSNR=%.4f\n',c.id,methods{m},item.Parameters,scores(p));
            end
            [~,best] = max(scores);
        else
            best = 1;
            % Initialize libraries and FFT/image operations on validation data
            % before timing tests, including fixed-parameter HQS methods.
            thesis_restore(valY{1},c,valMeta{1},methods{m},grids{m}{1},cfg.device);
        end
        selected{m} = grids{m}{best};
        selection.Case = c.id; selection.Method = methods{m};
        selection.Parameters = parameter_key(selected{m});
        selections = append_struct(selections,selection);
    end
    caseFolder = fullfile(outputFolder,c.id); mkdir(caseFolder);
    for j = 1:numel(testFiles)
        clean = read_clean(testFiles{j},c,cfg.cropSize);
        seed = image_seed(cfg.seed,c.id,c.dataset,testFiles{j},false);
        [y,meta] = thesis_degrade(clean,c,seed);
        [~,imageName,extension] = fileparts(testFiles{j});
        save(fullfile(caseFolder,[imageName '_observation.mat']),'clean','y','meta');
        if j <= cfg.exampleCount
            imwrite(im2uint8(clean),fullfile(caseFolder,[imageName '_Clean.png']));
            displayInput = y;
            if strcmp(c.task,'superresolution'), displayInput = imresize(y,c.scale,'bicubic'); end
            imwrite(im2uint8(displayInput),fullfile(caseFolder,[imageName '_Observation.png']));
        end
        for m = 1:numel(methods)
            if strcmp(cfg.device,'gpu'), wait(gpuDevice()); end
            start = tic;
            output = thesis_restore(y,c,meta,methods{m},selected{m},cfg.device);
            if strcmp(cfg.device,'gpu'), wait(gpuDevice()); end
            elapsed = toc(start);
            [psnrValue,ssimValue] = metrics(clean,output,c.border);
            row.Case = c.id; row.Task = c.task; row.Dataset = c.dataset;
            row.Image = [imageName extension]; row.Degradation = c.degradation;
            row.Level = c.level; row.Scale = c.scale; row.Channels = c.channels;
            row.Seed = seed; row.Method = methods{m}; row.Parameters = parameter_key(selected{m});
            row.PSNR = psnrValue; row.SSIM = ssimValue; row.Seconds = elapsed;
            row.Border = c.border; row.Height = size(clean,1); row.Width = size(clean,2);
            row.Execution = 'CPU'; row.KernelInformation = 'none';
            row.DenoiserModelIndex = 0;
            if strcmp(methods{m},'IRCNN'), row.Execution = ['MatConvNet-' upper(cfg.device)]; end
            if strcmp(methods{m},'IRCNN') && strcmp(c.task,'denoising')
                row.DenoiserModelIndex = min(25,max(1,ceil(selected{m}.sigma/2)));
            end
            if strcmp(c.task,'deblurring')
                row.KernelInformation = 'known';
                if strcmp(methods{m},'BlindDeconvolution'), row.KernelInformation = 'estimated'; end
                if strcmp(methods{m},'Input'), row.KernelInformation = 'none'; end
            end
            rows = append_struct(rows,row);
            if j <= cfg.exampleCount
                imwrite(im2uint8(output),fullfile(caseFolder,[imageName '_' methods{m} '.png']));
            end
        end
        fprintf('TEST %s %d/%d %s IRCNN_PSNR=%.4f\n', ...
            c.id,j,numel(testFiles),row.Image,rows(end-numel(methods)+2).PSNR);
        checkpoint(outputFolder,cfg,environment,rows,tuning,selections,complete);
    end
end
summary = summarize(rows);
writetable(struct2table(summary),fullfile(outputFolder,'summary.csv'));
write_report(outputFolder,cfg,summary);
thesis_plot_results(outputFolder);
environment.finished = datestr(now,30); complete = true;
checkpoint(outputFolder,cfg,environment,rows,tuning,selections,complete);
fprintf('THESIS_RUN_COMPLETE %s\n',outputFolder);
disp(struct2table(summary));
end

function files = image_files(root,dataset)
folder = fullfile(root,'testsets',dataset);
assert(exist(folder,'dir') == 7,'thesis:Dataset','Missing dataset: %s.',folder);
entries = dir(folder); files = {};
for k = 1:numel(entries)
    [~,~,ext] = fileparts(entries(k).name);
    if ~entries(k).isdir && any(strcmpi(ext,{'.png','.bmp','.jpg','.jpeg','.tif','.tiff'}))
        files{end+1} = fullfile(folder,entries(k).name); %#ok<AGROW>
    end
end
files = sort(files);
assert(~isempty(files),'thesis:Dataset','Dataset %s has no images.',dataset);
end

function clean = read_clean(path,c,cropSize)
clean = thesis_read_clean(path,c,cropSize);
end

function check_disjoint(testFiles,validationFiles,c)
% Compare processed full-image pixels, so grayscale/RGB duplicates are caught.
% SR may legitimately replicate grayscale sources. Skip only its scale crop
% during full-image hashing, while preserving strict RGB denoising validation.
if strcmp(c.task,'superresolution'), c.task='full_image_hash'; end
c.border = 0;
testHashes = cell(size(testFiles)); validationHashes = cell(size(validationFiles));
for k = 1:numel(testFiles), testHashes{k} = pixel_hash(read_clean(testFiles{k},c,Inf)); end
for k = 1:numel(validationFiles), validationHashes{k} = pixel_hash(read_clean(validationFiles{k},c,Inf)); end
assert(isempty(intersect(testHashes,validationHashes)),'thesis:DataLeakage', ...
    'A validation image is identical to a test image; choose a disjoint validation set.');
end

function seed = image_seed(base,cid,dataset,path,validation)
[~,name,ext] = fileparts(path); key = [cid '/' dataset '/' name ext];
hash = 0;
for value = double(key), hash = mod(hash*31+value,2147483647); end
seed = mod(base+hash+100000000*validation,4294967295);
end

function [p,s] = metrics(clean,output,border)
assert(isequal(size(clean),size(output)) && all(isfinite(output(:))), ...
    'thesis:Output','Restoration must have the reference dimensions and finite pixels.');
[p,s] = Cal_PSNRSSIM(im2uint8(clean),im2uint8(output),border,border);
assert(isfinite(p) && isfinite(s),'thesis:Metrics','Invalid PSNR/SSIM.');
end

function warm_models(c,device)
sigmas = 1:2:49;
if strcmp(c.task,'denoising') && strcmp(c.degradation,'gaussian'), sigmas = c.level; end
input = zeros(32,32,c.channels,'single');
if strcmp(device,'gpu'), input = gpuArray(input); end
for sigma = sigmas, thesis_denoise(input,sigma,'matconvnet'); end
if strcmp(device,'gpu'), wait(gpuDevice()); end
end

function key = parameter_key(parameters)
names = sort(fieldnames(parameters)); key = 'fixed';
if isempty(names), return; end
key = '';
for k = 1:numel(names)
    if k > 1, key = [key ';']; end %#ok<AGROW>
    key = [key names{k} '=' num2str(parameters.(names{k}),'%.10g')]; %#ok<AGROW>
end
end

function hash = file_hash(path)
fid = fopen(path,'rb'); assert(fid >= 0,'Cannot read file for hashing.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
md = java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid), md.update(fread(fid,1048576,'*uint8')); end
hash = digest_text(md);
end

function hash = pixel_hash(image)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(sprintf('%d,',size(image))));
md.update(typecast(image(:),'uint8'));
hash = digest_text(md);
end

function hash = digest_text(md)
bytes = typecast(md.digest(),'uint8');
hash = lower(reshape(dec2hex(bytes,2)',1,[]));
end

function snapshot_source(root,outputFolder)
destination = fullfile(outputFolder,'source'); mkdir(destination);
copyfile(fullfile(root,'Run_thesis_experiments.m'),destination);
copyfile(fullfile(root,'Setup_thesis_matconvnet.m'),destination);
for folder = {'experiments','utilities'}
    target = fullfile(destination,folder{1}); mkdir(target);
    copyfile(fullfile(root,folder{1},'*.m'),target);
end
mkdir(fullfile(destination,'experiments','setup'));
copyfile(fullfile(root,'experiments','setup','cudnn8_compat.hpp'), ...
    fullfile(destination,'experiments','setup'));
end

function checkpoint(folder,cfg,environment,rows,tuning,selections,complete)
save(fullfile(folder,'state.mat'),'cfg','environment','rows','tuning','selections','complete');
if ~isempty(rows), writetable(struct2table(rows),fullfile(folder,'per_image.csv')); end
if ~isempty(tuning), writetable(struct2table(tuning),fullfile(folder,'validation.csv')); end
if ~isempty(selections), writetable(struct2table(selections),fullfile(folder,'parameters.csv')); end
end

function output = summarize(rows)
output = struct([]);
cases = unique({rows.Case},'stable');
for k = 1:numel(cases)
    caseRows = rows(strcmp({rows.Case},cases{k}));
    methods = unique({caseRows.Method},'stable');
    for m = 1:numel(methods)
        selected = caseRows(strcmp({caseRows.Method},methods{m}));
        row.Case = cases{k}; row.Method = methods{m}; row.N = numel(selected);
        row.MeanPSNR = mean([selected.PSNR]); row.StdPSNR = std([selected.PSNR]);
        row.MeanSSIM = mean([selected.SSIM]); row.StdSSIM = std([selected.SSIM]);
        row.MeanSeconds = mean([selected.Seconds]);
        row.Execution = selected(1).Execution;
        row.KernelInformation = selected(1).KernelInformation;
        ircnn = caseRows(strcmp({caseRows.Method},'IRCNN'));
        assert(isequal({ircnn.Image},{selected.Image}),'Unpaired summary observations.');
        delta = [ircnn.PSNR]-[selected.PSNR];
        row.IRCNNMinusMethodPSNR = mean(delta);
        row.IRCNNWins = sum(delta > 0);
        output = append_struct(output,row);
    end
end
end

function list = append_struct(list,item)
if isempty(list), list = item;
else list(end+1) = item; end
end

function write_report(folder,cfg,summary)
fid = fopen(fullfile(folder,'REPORT.md'),'w'); assert(fid >= 0,'Cannot write report.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'# IRCNN thesis experiments: %s\n\n',cfg.profile);
if cfg.smokeOnly
    fprintf(fid,'**SMOKE ONLY: one image per case, top-left 64-pixel crops. These are software checks, not thesis measurements.**\n\n');
end
fprintf(fid,'The supplied pretrained models were evaluated with compiled MatConvNet on %s. No training was performed.\n\n',upper(cfg.device));
fprintf(fid,'| Condition | Method | N | Mean PSNR (dB) | Mean SSIM | Mean seconds | Execution |\n');
fprintf(fid,'|---|---|---:|---:|---:|---:|---|\n');
for k = 1:numel(summary)
    r = summary(k);
    fprintf(fid,'| %s | %s | %d | %.4f | %.6f | %.6f | %s |\n', ...
        r.Case,r.Method,r.N,r.MeanPSNR,r.MeanSSIM,r.MeanSeconds,r.Execution);
end
fprintf(fid,['\nEach method received the same saved floating-point observation per image and condition. ' ...
    'Gaussian and Poisson observations were not clipped before restoration. ' ...
    'Outputs were clipped and quantized to uint8 equally before the repository PSNR/SSIM functions. ' ...
    'RGB PSNR uses combined RGB MSE and SSIM averages channels; these are not luminance-only scores. ' ...
    'Denoising/deblurring use border 0; super-resolution uses border equal to scale.\n\n']);
fprintf(fid,['Parameters with multiple candidates were selected by mean validation PSNR on a separate local dataset. ' ...
    'Exact processed-pixel overlap was checked before tuning. This does not establish independence from the original authors'' model-training data. ' ...
    'The validation candidate scores and final selections are saved separately. ' ...
    'RNG states were restored after degradation generation; saved seeds reproduce paired observations.\n\n']);
fprintf(fid,['Times include restoration, CPU/GPU transfers, and GPU synchronization; ' ...
    'they exclude initial model loading, warm-up, image I/O, and metrics. Baselines run on CPU. ' ...
    'Hardware differs by method, so these timings describe this installation and do not isolate algorithmic complexity. ' ...
    'PSNR/SSIM standard deviations are across test images; one deterministic noise realization was used per image.\n\n']);
fprintf(fid,['The original Gaussian denoisers applied to Poisson or impulse observations are robustness experiments, ' ...
    'not models trained for those noise distributions. Wiener uses known Gaussian variance or an observation-based Poisson estimate; ' ...
    'the impulse nominal variance is p/6. Median and bilateral filters use symmetric padding; bilateral filtering is per channel. ' ...
    'Wiener2 uses MATLAB''s standard local-statistics boundary handling.\n\n']);
fprintf(fid,['Deblurring uses periodic convolution. IRCNN, Wiener deconvolution, and Richardson-Lucy receive the true PSF. ' ...
    'Blind deconvolution receives only a fixed 25x25 Gaussian initialization (sigma 2); ' ...
    'it therefore has less information and is labelled estimated-kernel in the CSV. ' ...
    'MATLAB deconvblind also has its own boundary behavior. Gaussian observation noise does not match the Poisson likelihood of Richardson-Lucy. ' ...
    'Compare these with those limitations stated. IRCNN uses the demo''s 30-iteration 49-to-13 schedule and lambda=noise variance/3.\n\n']);
fprintf(fid,['Super-resolution follows Demo_SISR_RGB''s iterative back-projection with 30 outer iterations, ' ...
    '5 inner steps, step size 1.75, and denoiser sigma 12*scale down to scale. ' ...
    'LR observations are quantized to 8 bits. Fourier is periodic band-limited interpolation with sampling-grid alignment; ' ...
    'it does not compensate Gaussian blur.\n\n']);
fprintf(fid,['Configuration, completion flag, per-image rows, selected parameters, SHA-256 model/binary fingerprints, ' ...
    'and code snapshots are in state.mat and source/. An interrupted run has complete=false. ' ...
    'The original IRCNN models and method must be attributed to Zhang et al., Learning Deep CNN Denoiser Prior for Image Restoration, CVPR 2017.\n']);
end
