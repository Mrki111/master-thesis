function thesis_plot_results(folder)
data = readtable(fullfile(folder,'summary.csv'));
cases = unique(data.Case,'stable');
for k = 1:numel(cases)
    selected = data(strcmp(data.Case,cases{k}),:);
    labels = selected.Method;
    labels(strcmp(labels,'WienerDeconvolution')) = {'Wiener deconv.'};
    labels(strcmp(labels,'RichardsonLucy')) = {'Richardson-Lucy'};
    labels(strcmp(labels,'BlindDeconvolution')) = {'Blind deconv.'};
    values = {selected.MeanPSNR,selected.MeanSSIM,selected.MeanSeconds};
    titles = {'Mean PSNR (dB)','Mean SSIM','Mean restoration time (s)'};
    figureHandle = figure('Visible','off','Color','w','Position',[50 50 1320 390]);
    cleanup = onCleanup(@() close(figureHandle));
    for panel = 1:3
        subplot(1,3,panel);
        barh(1:numel(labels),values{panel},.65,'FaceColor',[.25 .5 .75]); hold on;
        index = find(strcmp(selected.Method,'IRCNN'));
        barh(index,values{panel}(index),.65,'FaceColor',[.85 .3 .2]);
        set(gca,'YTick',1:numel(labels),'YTickLabel',labels, ...
            'YDir','reverse','FontSize',10,'Box','off');
        xlabel(titles{panel}); title(sprintf('N = %d images',selected.N(1)));
        grid on;
    end
    set(figureHandle,'PaperPositionMode','auto');
    print(figureHandle,fullfile(folder,[cases{k} '_summary.png']),'-dpng','-r120');
    clear cleanup;
end
end
