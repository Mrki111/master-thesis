function folder = thesis_parent_run(data,root)
% Resolve inherited observations in a relocated reproduction package first.
assert(isfield(data,'parentRun') && ischar(data.parentRun), ...
    'thesis:ParentRunMissing','The saved state has no parent-run reference.');
stored = regexprep(data.parentRun,'[\\/]+$','');
name = regexp(stored,'[^\\/]+$','match','once');
assert(~isempty(name),'thesis:ParentRunMissing','The saved parent-run reference is empty.');
candidates = {fullfile(root,'results','thesis',name),data.parentRun};
for k = 1:numel(candidates)
    if exist(fullfile(candidates{k},'state.mat'),'file') == 2
        folder = candidates{k}; return;
    end
end
error('thesis:ParentRunMissing', ...
    'Parent run %s is missing. Extract the complete reproduction archive.',name);
end
