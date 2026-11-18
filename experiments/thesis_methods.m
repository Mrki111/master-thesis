function [methods, grids] = thesis_methods(c, search)
% Every candidate is fixed before seeing test metrics.
% Keep the original grid reproducible; the review uses a separate expansion.
if nargin < 2, search = 'original'; end
assert(any(strcmp(search,{'original','expanded'})),'thesis:Grid','Unknown grid.');
switch c.task
    case 'denoising'
        methods = {'Input','IRCNN','Wiener','Median','Bilateral'};
        grids = {{struct()}, {}, {}, {}, {}};
        if strcmp(c.degradation,'gaussian'), sigmas = c.level;
        else sigmas = [5 15 25 35 50]; end
        for value = sigmas, grids{2}{end+1} = struct('sigma',value); end
        for window = [3 5 7]
            grids{3}{end+1} = struct('window',window);
            grids{4}{end+1} = struct('window',window);
        end
        for spatial = [1 2]
            for range = [.75 1.5 3]
                grids{5}{end+1} = struct('spatial',spatial,'range',range);
            end
        end
        if strcmp(search,'expanded')
            % Inf is the Gaussian spatial-filter limit, not a finite optimum.
            for spatial = [1 2 3]
                for range = [3 6 12 Inf]
                    if spatial <= 2 && range == 3, continue; end
                    grids{5}{end+1} = struct('spatial',spatial,'range',range);
                end
            end
        end
    case 'deblurring'
        methods = {'Input','IRCNN','WienerDeconvolution','RichardsonLucy','BlindDeconvolution'};
        grids = {{struct()}, {struct()}, {}, {}, {}};
        for multiplier = [.25 1 4], grids{3}{end+1} = struct('multiplier',multiplier); end
        for iterations = [10 20 40]
            grids{4}{end+1} = struct('iterations',iterations);
            grids{5}{end+1} = struct('iterations',iterations);
        end
        if strcmp(search,'expanded')
            for value = [16 64 256 1024], grids{3}{end+1} = struct('multiplier',value); end
            for value = [80 160 320], grids{4}{end+1} = struct('iterations',value); end
            for value = [1 2 5], grids{5}{end+1} = struct('iterations',value); end
        end
    case 'superresolution'
        methods = {'Bicubic','IRCNN','Fourier'};
        grids = {{struct()}, {struct()}, {struct()}};
end
end
