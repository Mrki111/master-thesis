function Verify_thesis_review(folder)
% Validate paired coverage, inherited rows, selection and the RGB audit.
root=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'experiments'));
if nargin<1,folder=fullfile(root,'results','thesis','mentor_review_20261118');end
d=load(fullfile(folder,'state.mat'));
assert(d.complete && numel(d.rows)==5084,'Incomplete supplemental coverage.');
assert(isfield(d.cfg,'bm3dGrayscaleGaussian') && d.cfg.bm3dGrayscaleGaussian);
Test_thesis_experiments(folder);
parentFolder=thesis_parent_run(d,root);
parent=load(fullfile(parentFolder,'state.mat'));
assert(numel(d.tuning)==numel(parent.tuning)+160,'Expanded validation candidate coverage changed.');
assert(strcmp(file_hash(fullfile(root,'models','modelgray.mat')),parent.environment.modelgraySHA256));
assert(strcmp(file_hash(fullfile(root,'models','modelcolor.mat')),parent.environment.modelcolorSHA256));
localGPUConvolution=fullfile(root,'matconvnet','matconvnet-1.0-beta25-gpu','matlab','mex','vl_nnconv.mexw64');
assert(strcmp(file_hash(localGPUConvolution),parent.environment.convolutionSHA256));
fourier=load(fullfile(root,'results','thesis','fourier_alignment_20261021','state.mat'));
bm=find(strcmp({d.rows.Method},'BM3D')); assert(numel(bm)==204);
assert(numel(d.changes)==20,'Missing parameter-grid review.');
changed=0; inherited=0;
for k=1:numel(fourier.rows)
    before=fourier.rows(k); after=d.rows(k);
    affected=any(strcmp({d.changes.Case},before.Case)&strcmp({d.changes.Method},before.Method));
    if ~affected || strcmp(before.Parameters,after.Parameters)
        assert(isequal(before,after),'An inherited row changed.'); inherited=inherited+1;
    else
        assert(before.Seed==after.Seed && strcmp(before.Image,after.Image)); changed=changed+1;
    end
end
a=load(fullfile(folder,'rgb_audit.mat'));
assert(numel(a.audit)==476 && numel(a.reproduced)==1428);
assert(max(abs([a.reproduced.PSNRError]))<1e-8 && max(abs([a.reproduced.SSIMError]))<1e-8);
for k=bm
    r=d.rows(k);
    match=parent.rows(strcmp({parent.rows.Case},r.Case)&strcmp({parent.rows.Image},r.Image)&strcmp({parent.rows.Method},'IRCNN'));
    assert(numel(match)==1 && match.Seed==r.Seed && r.Channels==1);
end
% Existing reviewed records are immutable during a later verification.
if exist(fullfile(folder,'VERIFIED.json'),'file')~=2
snapshot=fullfile(folder,'source');
for name={'Run_thesis_experiments.m','Setup_thesis_matconvnet.m', ...
        'Review_thesis_experiments.m','Verify_thesis_review.m','Test_thesis_experiments.m'}
    copyfile(fullfile(root,name{1}),snapshot);
end
copyfile(fullfile(root,'experiments','*.m'),fullfile(snapshot,'experiments'));
fid=fopen(fullfile(snapshot,'README.md'),'w'); assert(fid>=0);
fprintf(fid,['# Supplemental experiment source\n\nThis is the final review and verification source. ' ...
    'Unchanged inherited measurements retain the historical source snapshots in the parent full run ' ...
    'and Fourier correction. The restricted BM3D implementation is downloaded separately.\n']);
fclose(fid);
fid=fopen(fullfile(folder,'VERIFIED.json'),'w'); assert(fid>=0);
fprintf(fid,'{"rows":5084,"bm3d_rows":204,"rgb_observations":476,"rgb_reproduced":1428,"changed_baseline_rows":%d,"inherited_rows":%d,"checks_passed":true}\n',changed,inherited);
fclose(fid);
end
fprintf('MENTOR_REVIEW_VERIFIED rows=5084 changed=%d inherited=%d\n',changed,inherited);
end

function value=file_hash(path)
fid=fopen(path,'rb'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(fread(fid,Inf,'*uint8'));
bytes=typecast(md.digest(),'uint8'); value=lower(reshape(dec2hex(bytes,2)',1,[]));
end
