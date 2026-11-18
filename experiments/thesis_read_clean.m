function clean = thesis_read_clean(path,c,cropSize)
% Shared image loading; a grayscale file must not silently enter CBSD68 tests.
[data,map] = imread(path);
if ~isempty(map), data = ind2rgb(data,map); end
clean = im2single(data);
if c.channels == 1 && size(clean,3) == 3, clean = rgb2gray(clean); end
if c.channels == 3 && size(clean,3) == 1
    assert(~strcmp(c.task,'denoising'),'thesis:RGBInput', ...
        'RGB denoising requires an actual color input: %s',path);
    clean = repmat(clean,[1 1 3]);
end
assert(size(clean,3)==c.channels,'thesis:Image','Unsupported image channels.');
if isfinite(cropSize)
    clean=clean(1:min(size(clean,1),cropSize),1:min(size(clean,2),cropSize),:);
end
if strcmp(c.task,'superresolution'), clean=modcrop(clean,c.scale); end
assert(min(size(clean,1),size(clean,2))-2*c.border>=11, ...
    'thesis:Image','Image too small for SSIM with this border.');
end
