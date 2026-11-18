function y = upsample2(x,K)
y = zeros(size(x,1)*K,size(x,2)*K,size(x,3),'like',x);
y(1:K:end,1:K:end,:) = x;
end
