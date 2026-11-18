function info = Setup_thesis_matconvnet(doCompile, matconvnetRoot, device, cudaRoot, cudnnRoot, resumeBuild)
%SETUP_THESIS_MATCONVNET Set up the README's official MatConvNet backend.
% Setup_thesis_matconvnet(true) compiles the CPU MEX files once.
% Setup_thesis_matconvnet(false) only loads and verifies an existing build.
% Setup_thesis_matconvnet(true,[],'gpu') builds a separate GPU copy with cuDNN.
% Setup_thesis_matconvnet(false,[],'gpu') selects and checks that GPU build.
if nargin < 1, doCompile = false; end
if nargin < 3, device = 'cpu'; end
if nargin < 6, resumeBuild = false; end
assert(any(strcmpi(device,{'cpu','gpu'})),'thesis:Device','Use cpu or gpu.');
useGPU = strcmpi(device,'gpu');
root = fileparts(mfilename('fullpath'));
if nargin < 2 || isempty(matconvnetRoot)
    candidates = {getenv('MATCONVNET_ROOT'), ...
        fullfile(root,'matconvnet','matconvnet-1.0-beta25'), ...
        fullfile(matlabroot,'matconvnet','matconvnet-1.0-beta25')};
    matconvnetRoot = '';
    for k = 1:numel(candidates)
        if ~isempty(candidates{k}) && ...
                exist(fullfile(candidates{k},'matlab','vl_setupnn.m'),'file') == 2
            matconvnetRoot = candidates{k}; break;
        end
    end
    if useGPU
        gpuRoot = fullfile(root,'matconvnet','matconvnet-1.0-beta25-gpu');
        if doCompile && ~exist(fullfile(gpuRoot,'matlab','vl_setupnn.m'),'file')
            assert(~isempty(matconvnetRoot),'thesis:MatConvNetMissing','MatConvNet source is required.');
            mkdir(gpuRoot);
            % Copy source and MATLAB code, leaving the working CPU binaries in place.
            copyfile(fullfile(matconvnetRoot,'COPYING'),gpuRoot);
            mkdir(fullfile(gpuRoot,'matlab'));
            entries = dir(fullfile(matconvnetRoot,'matlab'));
            for j = 1:numel(entries)
                if any(strcmp(entries(j).name,{'.','..','mex'})), continue; end
                copyfile(fullfile(matconvnetRoot,'matlab',entries(j).name), ...
                    fullfile(gpuRoot,'matlab',entries(j).name));
            end
        end
        if doCompile && ~exist(fullfile(gpuRoot,'examples'),'dir')
            copyfile(fullfile(matconvnetRoot,'examples'),fullfile(gpuRoot,'examples'));
        end
        matconvnetRoot = gpuRoot;
    end
end
assert(exist(fullfile(matconvnetRoot,'matlab','vl_setupnn.m'),'file') == 2, ...
    'thesis:MatConvNetMissing','Set MATCONVNET_ROOT or pass the MatConvNet folder.');
if useGPU
    if nargin < 4 || isempty(cudaRoot)
        cudaRoot = getenv('CUDA_PATH');
        if isempty(cudaRoot)
            cudaRoot = fullfile(getenv('ProgramFiles'),'NVIDIA GPU Computing Toolkit','CUDA','v11.0');
        end
    end
    if nargin < 5 || isempty(cudnnRoot)
        cudnnRoot = cudaRoot;
        if ~exist(fullfile(cudnnRoot,'include','cudnn.h'),'file')
            cudnnRoot = fullfile(cudaRoot,'cuda');
        end
    end
    assert(exist(fullfile(cudaRoot,'bin','nvcc.exe'),'file') == 2, ...
        'thesis:CUDA','CUDA nvcc.exe was not found. Pass the CUDA root as argument 4.');
    assert(exist(fullfile(cudnnRoot,'include','cudnn.h'),'file') == 2, ...
        'thesis:CuDNN','cuDNN headers were not found. Pass the cuDNN root as argument 5.');
    setenv('PATH',[fullfile(cudnnRoot,'bin') ';' fullfile(cudaRoot,'bin') ';' getenv('PATH')]);
end
clear vl_nnconv vl_nnconvt vl_nnpool vl_nnbnorm vl_nnrelu vl_rootnn vl_setupnn vl_compilenn;
addpath(fullfile(matconvnetRoot,'matlab'));
if doCompile
    options = {'enableGpu',useGPU,'enableImreadJpeg',false,'verbose',1, ...
        'continue',~useGPU || resumeBuild};
    if useGPU
        % Embed PTX for forward compatibility with Ada on CUDA 11.0.
        options = [options {'enableCudnn',true,'cudaMethod','nvcc', ...
            'cudaRoot',cudaRoot,'cudnnRoot',cudnnRoot,'cudaArch', ...
            '-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75'}];
    end
    if ispc
        vcvars = fullfile(getenv('ProgramFiles(x86)'), ...
            'Microsoft Visual Studio 14.0','VC','vcvarsall.bat');
        assert(exist(vcvars,'file') == 2,'thesis:CompilerMissing', ...
            'Install the Visual Studio 2015 C++ tools and Windows SDK.');
        [status, environment] = system(['call "' vcvars '" amd64 >nul && set']);
        assert(status == 0,'thesis:CompilerEnvironment','vcvarsall failed.');
        lines = regexp(environment,'\r?\n','split');
        for k = 1:numel(lines)
            equal = find(lines{k} == '=',1);
            if isempty(equal) || equal == 1, continue; end
            name = lines{k}(1:equal-1);
            if any(strcmpi(name,{'PATH','INCLUDE','LIB','LIBPATH', ...
                    'VS140COMNTOOLS','VSINSTALLDIR','VCINSTALLDIR','WindowsSdkDir'}))
                setenv(name,lines{k}(equal+1:end));
            end
        end
    end
    % R2015a ships only a VS2013 XML. Keep the VS2015 adaptation local.
    if ispc && exist(fullfile(matlabroot,'bin','win64','mexopts','msvcpp2015.xml'),'file') ~= 2
        template = fullfile(matlabroot,'bin','win64','mexopts','msvcpp2013.xml');
        assert(exist(template,'file') == 2,'thesis:CompilerConfig','No MSVC template found.');
        text = fileread(template);
        text = strrep(text,'2013','2015');
        text = strrep(text,'MSVCPP120','MSVCPP140');
        text = strrep(text,'12.0','14.0');
        text = strrep(text,'VS120','VS140');
        % Search actual library folders first. App execution aliases in PATH
        % cannot be enumerated by MATLAB R2015a's older filesystem library.
        text = strrep(text, ...
            '$$LIB;$$LIBPATH;$$PATH;$$INCLUDE;$MATLABROOT\extern\lib\$ARCH\microsoft', ...
            '$MATLABROOT\extern\lib\$ARCH\microsoft;$$LIB;$$LIBPATH');
        % Use the verified compiler location, including Build Tools editions
        % which do not supply the Visual Studio IDE required by old finders.
        vcRoot = fileparts(vcvars);
        vsRoot = fileparts(vcRoot);
        sdkRoot = fullfile(getenv('ProgramFiles(x86)'),'Windows Kits','8.1');
        finderNames = {'VCROOT','SDKROOT','VSINSTALLDIR','VCINSTALLDIR','PROF_ENV','VCVARSALLDIR'};
        finderValues = {vcRoot,sdkRoot,vsRoot,vcRoot,fullfile(vsRoot,'Common7','Tools'),vcRoot};
        newFinder = '<locationFinder>';
        for k = 1:numel(finderNames)
            variable = ['THESIS_' finderNames{k}];
            setenv(variable,finderValues{k});
            newFinder = [newFinder sprintf('<%s><envVarExists name="%s" /></%s>', ...
                finderNames{k},variable,finderNames{k})]; %#ok<AGROW>
        end
        newFinder = [newFinder '</locationFinder>'];
        finderStart = strfind(text,'<locationFinder>');
        finderEnd = strfind(text,'</locationFinder>')+numel('</locationFinder>')-1;
        text = strrep(text,text(finderStart:finderEnd),newFinder);
        envStart = strfind(text,'    <env');
        envEnd = strfind(text(envStart:end),'/>');
        previousEnvironment = text(envStart:envStart+envEnd(1));
        inheritedEnvironment = '<env';
        environmentNames = {'PATH','INCLUDE','LIB','LIBPATH'};
        for k = 1:numel(environmentNames)
            value = getenv(environmentNames{k});
            value = strrep(strrep(value,'&','&amp;'),'"','&quot;');
            inheritedEnvironment = [inheritedEnvironment sprintf(' %s="%s"', ...
                environmentNames{k},value)]; %#ok<AGROW>
        end
        inheritedEnvironment = [inheritedEnvironment ' />'];
        text = strrep(text,previousEnvironment,inheritedEnvironment);
        configDir = fullfile(root,'experiments','setup');
        if ~exist(configDir,'dir'), mkdir(configDir); end
        compilerConfig = fullfile(configDir,'msvcpp2015_for_r2015a.xml');
        fid = fopen(compilerConfig,'w');
        assert(fid ~= -1,'Cannot write local compiler configuration.');
        cleaner = onCleanup(@() fclose(fid));
        fwrite(fid,text,'char');
        clear cleaner;
        mex(['-setup:' compilerConfig],'C++');
        options = [options {'mexConfig',compilerConfig,'preCompileFn', ...
            @(o,m,l,f) link_config(o,m,l,f,compilerConfig,root)}];
    end
    if useGPU && ~any(strcmp(options,'preCompileFn'))
        options = [options {'preCompileFn',@(o,m,l,f) link_config(o,m,l,f,'',root)}];
    end
    vl_compilenn(options{:});
end
vl_setupnn;
assert(exist('vl_nnconv','file') == 3,'thesis:MatConvNetUncompiled', ...
    ['MatConvNet source is installed but its MEX files are missing. ' ...
     'Run Setup_thesis_matconvnet(true) after configuring a C++ compiler.']);
actual = vl_nnconv(single(ones(8,8)),single(ones(3,3)),single(0),'pad',1);
assert(isequal(size(actual),[8 8]) && actual(4,4) == 9, ...
    'thesis:MatConvNetCheck','MatConvNet convolution failed its startup check.');
if useGPU
    gpuDevice();
    actual = gather(vl_nnconv(gpuArray(single(ones(8,8))), ...
        gpuArray(single(ones(3,3))),gpuArray(single(0)),'pad',1,'CuDNN'));
    assert(actual(4,4) == 9,'thesis:GPUCheck','MatConvNet GPU convolution check failed.');
end
info.root = matconvnetRoot;
info.convolutionBinary = which('vl_nnconv');
info.matlabVersion = version;
info.device = upper(device);
if useGPU
    selectedGPU = gpuDevice();
    info.gpuName = selectedGPU.Name;
    info.computeCapability = selectedGPU.ComputeCapability;
    info.driverVersion = selectedGPU.DriverVersion;
    info.matlabToolkitVersion = selectedGPU.ToolkitVersion;
    info.cudaRoot = cudaRoot;
    info.cudnnRoot = cudnnRoot;
end
fprintf('MATCONVNET_READY %s\n',info.convolutionBinary);
end

function [opts,mexSources,libSources,flags] = link_config(opts,mexSources,libSources,flags,config,root)
% Old MATLAB cannot infer a C++ linker from a list of object files alone.
if ~isempty(config), flags.mexlink = [flags.mexlink {'-f',config}]; end
for k = 1:numel(flags.mexlink)
    if strncmp(flags.mexlink{k},'-L',2)
        flags.mexlink{k} = strrep(flags.mexlink{k},'"','');
    end
end
if opts.enableCudnn
    % Keep the compatibility adapter outside the original MatConvNet sources.
    header = fullfile(root,'experiments','setup','cudnn8_compat.hpp');
    flags.nvcc = [flags.nvcc {['--pre-include="' header '"'], ...
        ['-I"' opts.cudnnIncludeDir '"']}];
end
if opts.enableGpu
    % CUDA 11 requires an explicit optimization level; MATLAB MEX uses -O.
    flags.base(strcmp(flags.base,'-O')) = [];
    flags.nvcc = [flags.nvcc {'-O3'}];
    flags.mex = [flags.mex {'-O'}];
    flags.mexlink = [flags.mexlink {'-O'}];
end
end
