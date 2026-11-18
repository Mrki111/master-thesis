function output = thesis_restore(y, c, meta, method, parameters, device)
% Restoration has no access to a clean test image or test metrics.
if strcmp(method,'Input'), output = y; return; end
if strcmp(method,'BM3D')
    assert(strcmp(c.task,'denoising') && strcmp(c.degradation,'gaussian') ...
        && size(y,3)==1,'thesis:BM3DScope','BM3D review is grayscale Gaussian only.');
    assert(exist('BM3D','file')==2,'thesis:BM3DMissing', ...
        'Download the official legacy BM3D v2.01 package; see REPRODUCIBILITY.md.');
    % Scalar 1 explicitly withholds the clean reference. Sigma is in 0..255 units.
    [~,output] = BM3D(1,double(y),c.level,'np',0);
    output = single(output);
    assert(isequal(size(output),size(y)) && all(isfinite(output(:))), ...
        'thesis:BM3DOutput','Invalid BM3D output.');
    return;
end
if strcmp(method,'Bicubic'), output = imresize(y,c.scale,'bicubic'); return; end
if strcmp(method,'Fourier')
    rows = size(y,1)*c.scale; cols = size(y,2)*c.scale;
    output = interpft(interpft(double(y),rows,1),cols,2);
    % Both LR grids are centred at (scale-1)/2 HR pixels. For the even
    % Gaussian kernel, imfilter contributes the additional half-pixel offset.
    shift = (c.scale-1)/2;
    ky = [0:floor(rows/2) -ceil(rows/2)+1:-1]';
    kx = [0:floor(cols/2) -ceil(cols/2)+1:-1];
    phase = exp(-2i*pi*shift*bsxfun(@plus,ky/rows,kx/cols));
    output = real(ifft2(bsxfun(@times,fft2(output),phase)));
    output = single(output); return;
end
if strcmp(method,'IRCNN')
    z = single(y);
    if strcmp(device,'gpu'), z = gpuArray(z); end
    switch c.task
        case 'denoising'
            z = thesis_denoise(z,parameters.sigma,'matconvnet');
        case 'deblurring'
            H = psf2otf(c.kernel,[size(y,1) size(y,2)]);
            numerator = bsxfun(@times,conj(H),fft2(double(y)));
            denominator = abs(H).^2;
            if strcmp(device,'gpu')
                numerator = gpuArray(single(numerator)); denominator = gpuArray(single(denominator));
            else numerator = single(numerator); denominator = single(denominator); end
            schedule = logspace(log10(49),log10(13),c.iterations);
            lambda = meta.noiseVariance/3;
            for iteration = 1:c.iterations
                rho = lambda*255^2/schedule(iteration)^2;
                z = real(ifft2(bsxfun(@rdivide,numerator+rho*fft2(z),denominator+rho)));
                z = thesis_denoise(z,schedule(iteration),'matconvnet');
            end
        case 'superresolution'
            z = imresize(z,c.scale,'bicubic');
            schedule = logspace(log10(12*c.scale),log10(c.scale),c.iterations);
            for iteration = 1:c.iterations
                for inner = 1:5
                    z = z+single(1.75)*imresize(single(y)- ...
                        imresize_down(z,c.scale,c.degradation,1.6),c.scale,'bicubic');
                end
                z = thesis_denoise(z,schedule(iteration),'matconvnet');
            end
    end
    if isa(z,'gpuArray'), z = gather(z); end
    output = single(z); return;
end
output = zeros(size(y),'single');
for channel = 1:size(y,3)
    plane = double(y(:,:,channel));
    switch method
        case 'Wiener'
            result = wiener2(plane,[parameters.window parameters.window],meta.noiseVariance);
        case 'Median'
            result = medfilt2(plane,[parameters.window parameters.window],'symmetric');
        case 'Bilateral'
            result = bilateral(plane,parameters.spatial, ...
                parameters.range*sqrt(max(meta.noiseVariance,eps)));
        case 'WienerDeconvolution'
            H = psf2otf(c.kernel,size(plane));
            signalVariance = max(var(plane(:))-meta.noiseVariance,eps);
            nsr = parameters.multiplier*meta.noiseVariance/signalVariance;
            result = real(ifft2(conj(H).*fft2(plane)./(abs(H).^2+nsr)));
        case 'RichardsonLucy'
            H = psf2otf(c.kernel,size(plane));
            result = max(plane,eps);
            positive = max(plane,0);
            for iteration = 1:parameters.iterations
                prediction = max(real(ifft2(H.*fft2(result))),eps);
                correction = real(ifft2(conj(H).*fft2(positive./prediction)));
                result = max(result.*correction,0);
            end
        case 'BlindDeconvolution'
            % Unknown kernel values: fixed 25x25 Gaussian initialization.
            initialPSF = fspecial('gaussian',25,2);
            [result,~] = deconvblind(max(plane,0),initialPSF,parameters.iterations);
        otherwise
            error('thesis:Method','Unknown method %s.',method);
    end
    output(:,:,channel) = single(result);
end
assert(all(isfinite(output(:))),'thesis:Restoration','Nonfinite baseline output.');
end

function result = bilateral(input,spatial,range)
radius = ceil(3*spatial);
padded = padarray(input,[radius radius],'symmetric');
numerator = zeros(size(input)); denominator = zeros(size(input));
for dy = -radius:radius
    for dx = -radius:radius
        neighbor = padded(radius+1+dy:radius+size(input,1)+dy, ...
            radius+1+dx:radius+size(input,2)+dx);
        weight = exp(-(dy^2+dx^2)/(2*spatial^2)-(neighbor-input).^2/(2*range^2));
        numerator = numerator+weight.*neighbor;
        denominator = denominator+weight;
    end
end
result = numerator./denominator;
end
