function y = thesis_conv(x, weights, bias, pad, dilate, stride)
%THESIS_CONV Tiled cross-correlation with MatConvNet's zero-padding convention.
% Matrix multiplication uses MATLAB's BLAS. No learning or weight conversion.
if nargin < 6, stride = 1; end
if numel(pad) == 1, pad = repmat(pad,1,4); end
if numel(dilate) == 1, dilate = [dilate dilate]; end
if numel(stride) == 1, stride = [stride stride]; end
x = single(x); weights = single(weights);
[h,w,channels] = size(x);
[fh,fw,fc,outputs] = size(weights);
assert(channels == fc, 'thesis:Channels', 'Input/filter channel counts differ.');
padded = zeros(h+pad(1)+pad(2),w+pad(3)+pad(4),channels,'single');
padded(pad(1)+(1:h),pad(3)+(1:w),:) = x;
ph = size(padded,1); pw = size(padded,2);
oh = floor((ph-(fh-1)*dilate(1)-1)/stride(1))+1;
ow = floor((pw-(fw-1)*dilate(2)-1)/stride(2))+1;
assert(oh > 0 && ow > 0, 'thesis:FilterSize', 'Filter exceeds input.');
% Offsets follow MATLAB's filter layout: row, column, input channel.
[fr,fwc,ic] = ndgrid(0:fh-1,0:fw-1,0:channels-1);
offsets = fr(:)*dilate(1)+fwc(:)*dilate(2)*ph+ic(:)*ph*pw;
filters = reshape(weights,[],outputs);
yflat = zeros(oh*ow,outputs,'single');
tileSize = 4096;
for first = 1:tileSize:oh*ow
    locations = first:min(first+tileSize-1,oh*ow);
    base = mod(locations-1,oh)*stride(1)+floor((locations-1)/oh)*stride(2)*ph+1;
    patches = padded(bsxfun(@plus,offsets,base));
    values = patches.'*filters;
    if ~isempty(bias), values = bsxfun(@plus,values,reshape(single(bias),1,[])); end
    yflat(locations,:) = values;
end
y = reshape(yflat,oh,ow,outputs);
end
