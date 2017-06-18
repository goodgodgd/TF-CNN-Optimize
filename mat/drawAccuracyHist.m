function drawAccuracyHist()
clc
clear
% verbResult: datind, netind, classind, raw_acc, opt_acc
% [meanResults, verbResult] = evaluateCNNs();
% save([pwd, '/../../data/result.mat'], 'verbResult');

data = load([pwd, '/../../data/result.mat']);
perform = data.verbResult;

for netind=1:4
    data = perform(perform(:,2)==netind,:);
    drawHistForCNN(data(:,[1 4 5]));
    waitforbuttonpress
end
end


function drawHistForCNN(data)
datcol = 1;
befcol = 2;
aftcol = 3;
edges = 0:0.05:1;
centers = edges(1:end-1) + 0.025;
befpos = centers - 0.12;
aftpos = centers + 0.12;

figure(1)
for datind=1:3
    subplot(3,1,datind)
    subdata = data(data(:,datcol)==datind,:);
    befcounts = histcounts(subdata(:,befcol), edges);
    aftcounts = histcounts(subdata(:,aftcol), edges);
    bar(centers, [befcounts; aftcounts]');
end
end