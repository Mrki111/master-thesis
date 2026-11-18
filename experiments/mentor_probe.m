function mentor_probe
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'experiments'),fullfile(root,'utilities'));
d = load(fullfile(root,'results','thesis','full_20261011T180756','state.mat'));
disp(d.environment);
disp(d.environment.backend);
rgb = imread(fullfile(root,'testsets','CBSD68','101085.png'));
fprintf('RGB_SHAPE %s CHANNEL_DIFF %g %g\n',mat2str(size(rgb)), ...
    max(max(abs(double(rgb(:,:,1))-double(rgb(:,:,2))))), ...
    max(max(abs(double(rgb(:,:,2))-double(rgb(:,:,3))))));
addpath(fullfile(root,'experiments','third_party','bm3d'));
y = im2single(imread(fullfile(root,'testsets','BSD68','test001.png')));
[~,out] = BM3D(1,double(y(1:64,1:64)),25,'np',0);
fprintf('BM3D_OK shape=%s finite=%d\n',mat2str(size(out)),all(isfinite(out(:))));
end
