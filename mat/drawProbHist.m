clc
clear

if ~isempty(strfind(pwd, '\CILAB_MACHINE'))
    datadir = 'C:\Users\CILAB_MACHINE\Desktop\CHD\easy-deep-paper\output-data';
else
    datadir = '/home/cideep/Work/tensorflow/output-data';
end
networks = {'inception_resnet_v2', 'inception_v4', 'resnet_v2_50', 'resnet_v2_101'};
datasets = {'cifar10', 'cifar100', 'voc2012'};
funcs = utilFuncs();

for datind=1:3
    trueProbs = [];
    for netind=1:4
        dirPath = [datadir, '/', cell2mat(networks(netind)), '_', cell2mat(datasets(datind))];
        [labels, probs] = funcs.loadData(dirPath, 'test');
        datalen = length(labels);
        probGTInds = sub2ind(size(probs), (1:datalen)', labels);
        gtprobs = probs(probGTInds);
        
        if netind==1
            trueProbs = gtprobs';
        else
            trueProbs = [trueProbs; gtprobs'];
        end
    end
    size(trueProbs)
    
    figure(1)
    bins = 0:0.05:1;
    histogram(trueProbs, bins)
%     set(gca,'YScale','log')
%     axis([0 1 0 20])
    waitforbuttonpress

end
