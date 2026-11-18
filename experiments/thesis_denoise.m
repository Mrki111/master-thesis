function [output, modelIndex] = thesis_denoise(input, sigma, backend)
%THESIS_DENOISE Evaluate the original, supplied IRCNN residual denoiser.
% Native backend supports the conv/relu layers in modelgray/modelcolor.mat.
if nargin < 3, backend = 'matconvnet'; end
persistent grayModels colorModels grayNets colorNets gpuGrayNets gpuColorNets
root = fileparts(fileparts(mfilename('fullpath')));
channels = size(input,3);
if channels == 1
    if isempty(grayModels)
        data = load(fullfile(root,'models','modelgray.mat'),'CNNdenoiser');
        grayModels = data.CNNdenoiser;
    end
    models = grayModels;
elseif channels == 3
    if isempty(colorModels)
        data = load(fullfile(root,'models','modelcolor.mat'),'CNNdenoiser');
        colorModels = data.CNNdenoiser;
    end
    models = colorModels;
else
    error('thesis:Channels','Only grayscale and RGB images are supported.');
end
assert(isscalar(sigma) && isfinite(sigma) && sigma > 0 && sigma <= 50, ...
    'thesis:Sigma','Denoiser sigma must be in (0,50], in 8-bit units.');
modelIndex = min(25,max(ceil(sigma/2),1));
layers = models{modelIndex};
input = single(input);
if strcmp(backend,'matconvnet')
    assert(exist('vl_nnconv','file') == 3,'thesis:MatConvNet', ...
        'Compiled MatConvNet is required. Run Setup_thesis_matconvnet.');
    onGPU = isa(input,'gpuArray');
    if channels == 1
        if isempty(grayNets), grayNets = cell(1,25); end
        if isempty(gpuGrayNets), gpuGrayNets = cell(1,25); end
        if isempty(grayNets{modelIndex})
            net.layers = layers;
            grayNets{modelIndex} = vl_simplenn_tidy(net);
        end
        net = grayNets{modelIndex};
        if onGPU
            if isempty(gpuGrayNets{modelIndex})
                gpuGrayNets{modelIndex} = vl_simplenn_move(net,'gpu');
            end
            net = gpuGrayNets{modelIndex};
        end
    else
        if isempty(colorNets), colorNets = cell(1,25); end
        if isempty(gpuColorNets), gpuColorNets = cell(1,25); end
        if isempty(colorNets{modelIndex})
            net.layers = layers;
            colorNets{modelIndex} = vl_simplenn_tidy(net);
        end
        net = colorNets{modelIndex};
        if onGPU
            if isempty(gpuColorNets{modelIndex})
                gpuColorNets{modelIndex} = vl_simplenn_move(net,'gpu');
            end
            net = gpuColorNets{modelIndex};
        end
    end
    res = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    residual = res(end).x;
elseif strcmp(backend,'native')
    assert(~isa(input,'gpuArray'),'thesis:NativeGPU','Native diagnostic backend is CPU only.');
    residual = input;
    for k = 1:numel(layers)
        layer = layers{k};
        switch layer.type
            case 'conv'
                residual = thesis_conv(residual,layer.weights{1}, ...
                    layer.weights{2},layer.pad,layer.dilate,layer.stride);
            case 'relu'
                leak = 0;
                if isfield(layer,'leak'), leak = layer.leak; end
                residual = max(residual,0)+leak*min(residual,0);
            otherwise
                error('thesis:Layer','Unsupported pretrained layer: %s',layer.type);
        end
    end
else
    error('thesis:Backend','Unknown backend: %s',backend);
end
output = input-residual;
finite = all(isfinite(output(:)));
if isa(finite,'gpuArray'), finite = gather(finite); end
assert(isequal(size(output),size(input)) && finite, ...
    'thesis:Inference','IRCNN produced invalid output.');
end
