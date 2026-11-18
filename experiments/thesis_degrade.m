function [y, meta] = thesis_degrade(clean, c, seed)
% Generate once per image/condition; all methods receive this same observation.
previous = rng; cleanup = onCleanup(@() rng(previous)); %#ok<NASGU>
rng(seed,'twister'); meta.seed = seed; meta.case = c;
switch c.task
    case 'denoising'
        switch c.degradation
            case 'gaussian'
                y = clean+single(c.level/255)*single(randn(size(clean)));
                meta.noiseVariance = (c.level/255)^2;
            case 'poisson'
                y = single(poissrnd(double(clean)*c.level)/c.level);
                meta.noiseVariance = mean(max(double(y(:)),0))/c.level;
            case 'impulse'
                mask = rand(size(clean,1),size(clean,2));
                y = clean;
                for channel = 1:size(clean,3)
                    plane = y(:,:,channel);
                    plane(mask < c.level/2) = 0;
                    plane(mask >= c.level/2 & mask < c.level) = 1;
                    y(:,:,channel) = plane;
                end
                % Nominal variance used only for baseline parameter scaling.
                meta.noiseVariance = c.level/6;
        end
    case 'deblurring'
        y = imfilter(clean,c.kernel,'circular','conv')+ ...
            single(c.level/255)*single(randn(size(clean)));
        meta.noiseVariance = (c.level/255)^2;
    case 'superresolution'
        % Match the demos' 8-bit LR observation, shared by every method.
        y = im2single(im2uint8(imresize_down(clean,c.scale,c.degradation,1.6)));
        meta.noiseVariance = 0;
    otherwise
        error('thesis:Task','Unknown restoration task.');
end
end
