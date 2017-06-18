function findChangedSamples()
competingBound = [0.1 0.8];
power = 1.5;
padWeight = 3;

if ~isempty(strfind(pwd, '\CILAB_MACHINE'))
    datadir = 'C:\Users\CILAB_MACHINE\Desktop\CHD\easy-deep-paper\output-data';
else
    datadir = '/home/cideep/Work/tensorflow/output-data';
end

optSplit = 'validation';
networks = {'inception_resnet_v2', 'inception_v4', 'resnet_v2_50', 'resnet_v2_101'};
netname = cell2mat(networks(2));
datasets = {'cifar10', 'cifar100', 'voc2012'};

funcs = utilFuncs();

for datind=1:length(datasets)
    dataname = cell2mat(datasets(datind));
    dirPath = [datadir, '/', netname, '_', dataname]
    [testLabels, testProbs] = funcs.loadData(dirPath, 'test');
    testLabels = testLabels(1:5000);
    testProbs = testProbs(1:5000,:);
    
    [valiLabels, valiProbs] = funcs.loadData(dirPath, optSplit);
    H = optimizeWeightInRange(valiLabels, valiProbs, competingBound, power, padWeight);
    testProbsCorr = funcs.correctProbs(testProbs, H);
    
    [~, rawLabels] = max(testProbs, [], 2);
    [~, corLabels] = max(testProbsCorr, [], 2);
    
    newTruePositive = find(rawLabels~=corLabels & corLabels==testLabels)
    newFalsePositive = find(rawLabels~=corLabels & rawLabels==testLabels)
    
    filename = sprintf('%s/../../data/truepos_%s.txt', pwd, dataname);
    dlmwrite(filename, newTruePositive, 'delimiter', '\t')
    filename = sprintf('%s/../../data/falspos_%s.txt', pwd, dataname);
    dlmwrite(filename, newFalsePositive, 'delimiter', '\t')
end
end
