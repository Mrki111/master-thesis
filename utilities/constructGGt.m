function GGt = constructGGt(h,K,rows,cols)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Eigen-decomposition for super-resolution
% Stanley Chan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(K >= 1 && K == fix(K) && mod(rows,K) == 0 && mod(cols,K) == 0, ...
    'constructGGt:Dimensions','Image dimensions must be divisible by K.');
% G*Gt is circular and shift invariant on the low-resolution grid. Taking
% its impulse response avoids cropping or miscentering rectangular kernels.
[G,Gt] = defGGt(h,K);
impulse = zeros(rows/K,cols/K);
impulse(1,1) = 1;
eigenvalues = fft2(G(Gt(impulse)));
assert(max(abs(imag(eigenvalues(:)))) < 1e-8*max(1,max(abs(eigenvalues(:)))), ...
    'constructGGt:Adjoint','G*Gt must have real eigenvalues.');
GGt = max(real(eigenvalues),0);
