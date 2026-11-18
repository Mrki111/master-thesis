function [G,Gt] = defGGt(h,K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Operators for super-resolution
% Stanley Chan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G  = @(x) fdown(x,h,K);
Gt = @(x) upf(x,h,K);
end

function y = fdown(x,h,K)
tmp = imfilter(x,h,'circular');
y = downsample2(tmp,K);
end

function y = upf(x,h,K)
tmp = upsample2(x,K);
y = imfilter(tmp,h,'circular','conv');
% imfilter keeps the same anchor when it flips a kernel. Even-sized kernels
% need a one-pixel shift to obtain the transpose of the correlation operator.
y = circshift(y,[mod(size(h,1)+1,2),mod(size(h,2)+1,2),0]);
end
