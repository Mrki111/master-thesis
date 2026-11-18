function outputFolder = Reevaluate_thesis_fourier(parentFolder,outputFolder)
% Re-evaluate the corrected Fourier baseline on unchanged saved observations.
% All other rows are inherited; the original experiment is never overwritten.
root = fileparts(mfilename('fullpath'));
if nargin < 1, parentFolder = fullfile(root,'results','thesis','full_20261011T180756'); end
if nargin < 2, outputFolder = fullfile(root,'results','thesis','fourier_alignment_20261021'); end
assert(~strcmp(parentFolder,outputFolder),'Correction output must differ from parent run.');
assert(exist(outputFolder,'dir') ~= 7,'Choose a new correction output directory.');
addpath(fullfile(root,'experiments'),fullfile(root,'utilities'));
data = load(fullfile(parentFolder,'state.mat'));
assert(data.complete && numel(data.rows)==4880,'Expected the completed full experiment.');
original = data.rows; rows = original; changes = struct([]); controls = struct([]);
mkdir(outputFolder);
for scale = [2 3 4]
    for kind = {'bicubic','Gaussian'}
        id = sprintf('superresolution_Set5_%s_0_x%d',kind{1},scale);
        indices = find(strcmp({rows.Case},id)&strcmp({rows.Method},'Fourier'));
        assert(numel(indices)==5,'Expected all five SR images.');
        first = load_observation(parentFolder,rows(indices(1)));
        thesis_restore(first.y,first.meta.case,first.meta,'Fourier',struct(),'cpu');
        corrected = strcmp(kind{1},'Gaussian') && mod(scale,2)==0;
        if corrected, mkdir(fullfile(outputFolder,id)); end
        for index = indices
            observation = load_observation(parentFolder,rows(index));
            c = observation.meta.case;
            started = tic;
            output = thesis_restore(observation.y,c,observation.meta,'Fourier',struct(),'cpu');
            elapsed = toc(started);
            [p,s] = Cal_PSNRSSIM(im2uint8(observation.clean),im2uint8(output),c.border,c.border);
            record.Case = id; record.Image = rows(index).Image;
            record.OldPSNR = rows(index).PSNR; record.OldSSIM = rows(index).SSIM;
            record.NewPSNR = p; record.NewSSIM = s; record.NewSeconds = elapsed;
            if corrected
                rows(index).PSNR = p; rows(index).SSIM = s; rows(index).Seconds = elapsed;
                changes = append(changes,record);
                [~,base] = fileparts(rows(index).Image);
                imwrite(im2uint8(output),fullfile(outputFolder,id,[base '_Fourier.png']));
                save(fullfile(outputFolder,id,[base '_Fourier.mat']),'output');
            else
                assert(abs(p-record.OldPSNR)<1e-8 && abs(s-record.OldSSIM)<1e-8,'An unaffected Fourier condition changed.');
                controls = append(controls,record);
            end
        end
    end
end
assert(numel(changes)==10 && numel(controls)==20,'Correction/control coverage changed.');
for index=1:numel(rows)
    affected = strcmp(rows(index).Method,'Fourier') && ...
        any(strcmp(rows(index).Case,{'superresolution_Set5_Gaussian_0_x2','superresolution_Set5_Gaussian_0_x4'}));
    if ~affected, assert(isequal(rows(index),original(index)),'An inherited row changed.'); end
end
summary = summarize(rows);
writetable(struct2table(rows),fullfile(outputFolder,'per_image.csv'));
writetable(struct2table(summary),fullfile(outputFolder,'summary.csv'));
writetable(struct2table(changes),fullfile(outputFolder,'correction.csv'));
writetable(struct2table(controls),fullfile(outputFolder,'unchanged_controls.csv'));
copyfile(fullfile(parentFolder,'parameters.csv'),outputFolder);
copyfile(fullfile(parentFolder,'validation.csv'),outputFolder);
copyfile(fullfile(parentFolder,'source'),fullfile(outputFolder,'source'));
copyfile(fullfile(root,'experiments','thesis_restore.m'),fullfile(outputFolder,'source','experiments','thesis_restore.m'));
copyfile(fullfile(root,'Reevaluate_thesis_fourier.m'),fullfile(outputFolder,'source'));
copyfile(fullfile(root,'Test_thesis_experiments.m'),fullfile(outputFolder,'source'));
data.rows = rows; data.summary = summary; data.parentRun = parentFolder;
data.environment.fourierCorrectionDate = datestr(now,30);
data.environment.fourierCorrectionChangedRows = numel(changes);
data.environment.fourierCorrectionControlRows = numel(controls);
save(fullfile(outputFolder,'state.mat'),'-struct','data');
Test_thesis_experiments(outputFolder);
fid=fopen(fullfile(outputFolder,'COMPLETE.txt'),'w');
fprintf(fid,'Fourier alignment correction: 10 updated rows, 20 unchanged controls, 4870 inherited rows.\n');
fclose(fid);
fprintf('FOURIER_CORRECTION_COMPLETE %s\n',outputFolder);
end

function data = load_observation(folder,row)
[~,base] = fileparts(row.Image);
data = load(fullfile(folder,row.Case,[base '_observation.mat']));
assert(data.meta.seed==row.Seed,'Observation seed changed.');
end

function output = summarize(rows)
output = struct([]); cases = unique({rows.Case},'stable');
for k=1:numel(cases)
    caseRows=rows(strcmp({rows.Case},cases{k})); methods=unique({caseRows.Method},'stable');
    ircnn=caseRows(strcmp({caseRows.Method},'IRCNN'));
    for m=1:numel(methods)
        selected=caseRows(strcmp({caseRows.Method},methods{m}));
        row.Case=cases{k}; row.Method=methods{m}; row.N=numel(selected);
        row.MeanPSNR=mean([selected.PSNR]); row.StdPSNR=std([selected.PSNR]);
        row.MeanSSIM=mean([selected.SSIM]); row.StdSSIM=std([selected.SSIM]);
        row.MeanSeconds=mean([selected.Seconds]); row.Execution=selected(1).Execution;
        row.KernelInformation=selected(1).KernelInformation;
        assert(isequal({ircnn.Image},{selected.Image}),'Unpaired observations.');
        delta=[ircnn.PSNR]-[selected.PSNR];
        row.IRCNNMinusMethodPSNR=mean(delta); row.IRCNNWins=sum(delta>0);
        output=append(output,row);
    end
end
end

function list = append(list,item)
if isempty(list),list=item;else list(end+1)=item;end
end
