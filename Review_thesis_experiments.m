function folder = Review_thesis_experiments(stage,folder)
% Independent review of RGB processing, BM3D and validation-grid boundaries.
% Original results are read only. Run stages in order: audit, bm3d, tuning.
if nargin<1, stage='audit'; end
root=fileparts(mfilename('fullpath'));
if nargin<2, folder=fullfile(root,'results','thesis','mentor_review_20261118'); end
assert(exist(fullfile(folder,'VERIFIED.json'),'file')~=2,'thesis:FrozenReview', ...
    'This review is verified and preserved. Choose a new output folder.');
addpath(fullfile(root,'experiments'),fullfile(root,'utilities'));
parent=fullfile(root,'results','thesis','full_20261011T180756');
merged=fullfile(root,'results','thesis','fourier_alignment_20261021');
d=load(fullfile(parent,'state.mat')); assert(d.complete,'Parent incomplete.');
if ~exist(folder,'dir'), mkdir(folder); end
snapshot=fullfile(folder,'source');
if ~exist(snapshot,'dir')
    mkdir(snapshot); mkdir(fullfile(snapshot,'experiments'));
    copyfile(fullfile(root,'experiments','*.m'),fullfile(snapshot,'experiments'));
    copyfile(fullfile(root,'experiments','setup'),fullfile(snapshot,'experiments','setup'));
    copyfile(fullfile(root,'utilities'),fullfile(snapshot,'utilities'));
    % Do not copy the authors' restricted third-party software into artifacts.
    % Only our own .m files and setup headers belong in the snapshot.
end
switch stage
    case 'audit'
        audit_rgb(root,parent,d,folder);
    case 'bm3d'
        addpath(fullfile(root,'experiments','third_party','bm3d'));
        rows=struct([]);
        for level=[15 25 50]
            c=d.cfg.cases(strcmp({d.cfg.cases.id},sprintf('denoising_BSD68_gaussian_%d_x1',level)));
            old=d.rows(strcmp({d.rows.Case},c.id)&strcmp({d.rows.Method},'IRCNN'));
            first=load_observation(parent,old(1));
            thesis_restore(first.y,c,first.meta,'BM3D',struct(),'cpu');
            for k=1:numel(old)
                o=load_observation(parent,old(k)); started=tic;
                output=thesis_restore(o.y,c,o.meta,'BM3D',struct(),'cpu'); elapsed=toc(started);
                [p,s]=score(o.clean,output,c.border);
                r=old(k); r.Method='BM3D'; r.Parameters='profile=np;sigma=known';
                r.Execution='CPU'; r.DenoiserModelIndex=0;
                r.PSNR=p; r.SSIM=s; r.Seconds=elapsed; rows=append(rows,r);
                if k<=3
                    caseFolder=fullfile(folder,c.id); if ~exist(caseFolder,'dir'),mkdir(caseFolder);end
                    [~,name]=fileparts(r.Image); imwrite(im2uint8(output),fullfile(caseFolder,[name '_BM3D.png']));
                end
                fprintf('BM3D %s %d/%d PSNR=%.5f\n',c.id,k,numel(old),p);
                save(fullfile(folder,'bm3d_state.mat'),'rows','parent');
            end
        end
        assert(numel(rows)==204,'BM3D coverage changed.');
        writetable(struct2table(rows),fullfile(folder,'bm3d_per_image.csv'));
        bm3dSHA256=hash_file(fullfile(root,'experiments','third_party','BM3D.zip'));
        save(fullfile(folder,'bm3d_state.mat'),'rows','parent','bm3dSHA256');
        write_comparison(rows,d.rows,folder,'bm3d');
    case 'tuning'
        tuning=d.tuning; selections=d.selections; changes=struct([]); rows=d.rows;
        % Preserve the previously corrected Fourier comparison.
        f=load(fullfile(merged,'state.mat')); rows=f.rows;
        cases=d.cfg.cases(~strcmp({d.cfg.cases.task},'superresolution'));
        for ci=1:numel(cases)
            c=cases(ci); [methods,grids]=thesis_methods(c,'expanded');
            valFiles=image_files(root,c.validation);
            clean=cell(size(valFiles)); y=clean; meta=clean;
            for j=1:numel(valFiles)
                clean{j}=thesis_read_clean(valFiles{j},c,Inf);
                seed=image_seed(d.cfg.seed,c.id,c.validation,valFiles{j},true);
                [y{j},meta{j}]=thesis_degrade(clean{j},c,seed);
            end
            if strcmp(c.task,'denoising'), active=5; else active=3:5; end
            for m=active
                method=methods{m}; scores=zeros(size(grids{m}));
                for pi=1:numel(grids{m})
                    key=parameter_key(grids{m}{pi});
                    previous=find(strcmp({tuning.Case},c.id)&strcmp({tuning.Method},method)&strcmp({tuning.Parameters},key));
                    if ~isempty(previous)
                        assert(numel(previous)==1); scores(pi)=tuning(previous).MeanPSNR;
                    else
                        values=zeros(size(valFiles));
                        for j=1:numel(valFiles)
                            output=thesis_restore(y{j},c,meta{j},method,grids{m}{pi},'cpu');
                            values(j)=score(clean{j},output,c.border);
                        end
                        scores(pi)=mean(values); t.Case=c.id; t.Method=method; t.Parameters=key;
                        t.ValidationDataset=c.validation; t.N=numel(valFiles); t.MeanPSNR=scores(pi);
                        tuning=append(tuning,t);
                    end
                    fprintf('EXPANDED %s %s %s %.6f\n',c.id,method,key,scores(pi));
                end
                [bestScore,best]=max(scores); chosen=parameter_key(grids{m}{best});
                si=find(strcmp({selections.Case},c.id)&strcmp({selections.Method},method));
                record.Case=c.id; record.Method=method; record.OldParameters=selections(si).Parameters;
                record.NewParameters=chosen; record.ValidationPSNR=bestScore;
                record.OldValidationPSNR=max([d.tuning(strcmp({d.tuning.Case},c.id)&strcmp({d.tuning.Method},method)).MeanPSNR]);
                changes=append(changes,record); selections(si).Parameters=chosen;
                indices=find(strcmp({rows.Case},c.id)&strcmp({rows.Method},method));
                if ~strcmp(record.OldParameters,chosen)
                    first=load_observation(parent,rows(indices(1)));
                    thesis_restore(first.y,c,first.meta,method,grids{m}{best},'cpu');
                    for k=indices
                        o=load_observation(parent,rows(k)); started=tic;
                        output=thesis_restore(o.y,c,o.meta,method,grids{m}{best},'cpu'); elapsed=toc(started);
                        [p,s]=score(o.clean,output,c.border);
                        rows(k).PSNR=p; rows(k).SSIM=s; rows(k).Seconds=elapsed; rows(k).Parameters=chosen;
                        names=unique({rows(indices).Image},'stable');
                        if any(strcmp(rows(k).Image,names(1:min(3,numel(names)))))
                            caseFolder=fullfile(folder,c.id); if ~exist(caseFolder,'dir'),mkdir(caseFolder);end
                            [~,name]=fileparts(rows(k).Image); imwrite(im2uint8(output),fullfile(caseFolder,[name '_' method '.png']));
                        end
                    end
                end
                save(fullfile(folder,'tuning_state.mat'),'rows','tuning','selections','changes','parent','merged');
                writetable(struct2table(changes),fullfile(folder,'grid_changes.csv'));
                writetable(struct2table(tuning),fullfile(folder,'validation.csv'));
            end
        end
        bm=load(fullfile(folder,'bm3d_state.mat')); rows=[rows bm.rows];
        for cid=unique({bm.rows.Case},'stable')
            s.Case=cid{1}; s.Method='BM3D'; s.Parameters='profile=np;sigma=known';
            selections=append(selections,s);
        end
        cfg=d.cfg; cfg.searchGrid='expanded'; cfg.bm3dGrayscaleGaussian=true;
        environment=d.environment; environment.reviewMatlabVersion=version;
        environment.reviewed=datestr(now,30); parentRun=parent; complete=true;
        save(fullfile(folder,'state.mat'),'cfg','rows','tuning','selections','changes','environment','parentRun','complete');
        writetable(struct2table(rows),fullfile(folder,'per_image.csv'));
        writetable(struct2table(selections),fullfile(folder,'parameters.csv'));
        write_comparison(rows,d.rows,folder,'expanded');
    otherwise
        error('thesis:Stage','Use audit, bm3d or tuning.');
end
copyfile(fullfile(root,'Review_thesis_experiments.m'),snapshot);
fprintf('MENTOR_REVIEW_STAGE_COMPLETE %s %s\n',stage,folder);
end

function audit_rgb(root,parent,d,folder)
audit=struct([]); reproduced=struct([]);
for ci=find(strcmp({d.cfg.cases.task},'denoising')&[d.cfg.cases.channels]==3)
    c=d.cfg.cases(ci); rows=d.rows(strcmp({d.rows.Case},c.id));
    inputRows=rows(strcmp({rows.Method},'Input'));
    for k=1:numel(inputRows)
        o=load_observation(parent,inputRows(k)); original=thesis_read_clean(fullfile(root,'testsets',c.dataset,inputRows(k).Image),c,Inf);
        assert(isequal(o.clean,original),'Saved RGB reference differs from original image.');
        assert(size(o.clean,3)==3 && any(any(o.clean(:,:,1)~=o.clean(:,:,2))) ...
            && any(any(o.clean(:,:,2)~=o.clean(:,:,3))),'RGB reference is replicated grayscale.');
        regenerated=thesis_degrade(o.clean,c,o.meta.seed);
        assert(isequal(regenerated,o.y),'Saved observation is not reproducible.');
        methods={'Wiener','Median','Bilateral'};
        for m=1:3
            selected=rows(strcmp({rows.Image},inputRows(k).Image)&strcmp({rows.Method},methods{m}));
            parameters=parse_key(selected.Parameters);
            output=thesis_restore(o.y,c,o.meta,methods{m},parameters,'cpu');
            % Independently invoke the single-channel path and compare all planes.
            for ch=1:3
                plane=thesis_restore(o.y(:,:,ch),c,o.meta,methods{m},parameters,'cpu');
                assert(isequal(output(:,:,ch),plane),'Channel loop differs from single-channel execution.');
            end
            [p,s]=score(o.clean,output,0); qaRef=im2uint8(o.clean); qaOut=im2uint8(output);
            errors=double(qaRef)-double(qaOut); expected=10*log10(255^2/mean(errors(:).^2));
            channelSSIM=zeros(1,3);
            for ch=1:3,[~,channelSSIM(ch)]=Cal_PSNRSSIM(qaRef(:,:,ch),qaOut(:,:,ch),0,0);end
            assert(abs(p-expected)<1e-12 && abs(s-mean(channelSSIM))<1e-12,'RGB metric aggregation failed.');
            assert(abs(p-selected.PSNR)<1e-8 && abs(s-selected.SSIM)<1e-8,'Saved baseline score differs.');
            r.Case=c.id; r.Image=selected.Image; r.Method=methods{m}; r.PSNR=p; r.SSIM=s;
            r.PSNRError=p-selected.PSNR; r.SSIMError=s-selected.SSIM; reproduced=append(reproduced,r);
        end
        a.Case=c.id; a.Image=inputRows(k).Image; a.Channels=3;
        a.RGMax=max(max(abs(double(o.clean(:,:,1))-double(o.clean(:,:,2)))));
        a.GBMax=max(max(abs(double(o.clean(:,:,2))-double(o.clean(:,:,3)))));
        audit=append(audit,a);
        fprintf('RGB_AUDIT %s %d/%d\n',c.id,k,numel(inputRows));
    end
end
assert(numel(audit)==476 && numel(reproduced)==1428,'RGB audit coverage changed.');
writetable(struct2table(audit),fullfile(folder,'rgb_inputs.csv'));
writetable(struct2table(reproduced),fullfile(folder,'rgb_reproduced.csv'));
environment=d.environment; save(fullfile(folder,'rgb_audit.mat'),'audit','reproduced','environment');
end

function write_comparison(rows,original,folder,label)
summary=struct([]);
for cid=unique({rows.Case},'stable')
    current=rows(strcmp({rows.Case},cid{1}));
    ir=original(strcmp({original.Case},cid{1})&strcmp({original.Method},'IRCNN'));
    for method=unique({current.Method},'stable')
        selected=current(strcmp({current.Method},method{1}));
        assert(isequal({selected.Image},{ir.Image}),'Image pairs differ.');
        s.Case=cid{1}; s.Method=method{1}; s.N=numel(selected);
        s.MeanPSNR=mean([selected.PSNR]); s.StdPSNR=std([selected.PSNR]);
        s.MeanSSIM=mean([selected.SSIM]); s.StdSSIM=std([selected.SSIM]);
        s.MeanSeconds=mean([selected.Seconds]);
        delta=[ir.PSNR]-[selected.PSNR]; s.IRCNNMinusMethodPSNR=mean(delta);
        s.IRCNNWins=sum(delta>0); s.MinimumDelta=min(delta); summary=append(summary,s);
    end
end
writetable(struct2table(summary),fullfile(folder,[label '_summary.csv']));
end

function [p,s]=score(clean,output,border)
assert(isequal(size(clean),size(output)) && all(isfinite(output(:))));
[p,s]=Cal_PSNRSSIM(im2uint8(clean),im2uint8(output),border,border);
assert(isfinite(p)&&isfinite(s));
end
function o=load_observation(parent,row)
[~,name]=fileparts(row.Image); o=load(fullfile(parent,row.Case,[name '_observation.mat']));
assert(o.meta.seed==row.Seed,'Seed differs.');
end
function files=image_files(root,dataset)
files={}; entries=dir(fullfile(root,'testsets',dataset));
for k=1:numel(entries)
    [~,~,ext]=fileparts(entries(k).name);
    if ~entries(k).isdir && any(strcmpi(ext,{'.png','.bmp','.jpg','.jpeg','.tif','.tiff'}))
        files{end+1}=fullfile(root,'testsets',dataset,entries(k).name); %#ok<AGROW>
    end
end
files=sort(files); assert(~isempty(files));
end
function seed=image_seed(base,cid,dataset,path,validation)
[~,name,ext]=fileparts(path); key=[cid '/' dataset '/' name ext]; h=0;
for value=double(key),h=mod(h*31+value,2147483647);end
seed=mod(base+h+100000000*validation,4294967295);
end
function key=parameter_key(p)
names=sort(fieldnames(p)); values=cell(size(names));
for k=1:numel(names),values{k}=sprintf('%s=%g',names{k},p.(names{k}));end
key=strjoin(values,';'); if isempty(key),key='fixed';end
end
function p=parse_key(key)
p=struct(); parts=strsplit(key,';');
for k=1:numel(parts),pair=strsplit(parts{k},'=');p.(pair{1})=str2double(pair{2});end
end
function list=append(list,item)
if isempty(list),list=item;else list(end+1)=item;end
end
function value=hash_file(path)
fid=fopen(path,'rb'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(fread(fid,Inf,'*uint8'));
bytes=typecast(md.digest(),'uint8'); value=lower(reshape(dec2hex(bytes,2)',1,[]));
end
