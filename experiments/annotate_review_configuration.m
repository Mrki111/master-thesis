function annotate_review_configuration(folder)
% Annotate a completed checkpoint produced by the initially cached runner.
% This migration changes configuration metadata only; preserve the raw state.
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<1,folder=fullfile(root,'results','thesis','mentor_review_20261118');end
d=load(fullfile(folder,'state.mat'));
assert(d.complete && numel(d.rows)==5084 && numel(d.changes)==20);
assert(sum(strcmp({d.rows.Method},'BM3D'))==204);
assert(~isfield(d.cfg,'bm3dGrayscaleGaussian'),'Configuration is already annotated.');
copyfile(fullfile(folder,'state.mat'),fullfile(folder,'state_before_config_annotation.mat'));
beforeRows=d.rows; beforeTuning=d.tuning;
d.cfg.searchGrid='expanded'; d.cfg.bm3dGrayscaleGaussian=true;
for c=unique({d.rows(strcmp({d.rows.Method},'BM3D')).Case},'stable')
    s.Case=c{1}; s.Method='BM3D'; s.Parameters='profile=np;sigma=known';
    d.selections(end+1)=s;
end
d.environment.configurationAnnotation=datestr(now,30);
d.environment.configurationAnnotationNote=['The initially cached runner saved the original full-profile configuration. ' ...
    'Expanded-grid and BM3D flags and fixed BM3D parameter rows were derived from the completed rows. ' ...
    'No measurement, tuning score, seed or observation was changed. The pre-annotation state is preserved.'];
assert(isequal(beforeRows,d.rows) && isequal(beforeTuning,d.tuning));
save(fullfile(folder,'state.mat'),'-struct','d');
writetable(struct2table(d.selections),fullfile(folder,'parameters.csv'));
fprintf('REVIEW_CONFIGURATION_ANNOTATED measurements_unchanged=5084\n');
end
