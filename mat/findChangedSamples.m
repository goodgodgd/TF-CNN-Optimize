function findChangedSamples()
clc
clear

competingBound = [0.1 0.8];
power = 1.5;
padWeight = 3;

[probDir, imgDir] = getDirs();
optSplit = 'validation';
networks = {'inception_resnet_v2', 'inception_v4', 'resnet_v2_50', 'resnet_v2_101'};
netname = cell2mat(networks(2));
datasets = {'cifar10', 'cifar100', 'voc2012'};

funcs = utilFuncs();

for datind=1:length(datasets)
    dataname = cell2mat(datasets(datind));
    dirPath = [probDir, '/', netname, '_', dataname];
    [testLabels, testProbs] = funcs.loadData(dirPath, 'test');
    testLabels = testLabels(1:5000);
    testProbs = testProbs(1:5000,:);
    
    [valiLabels, valiProbs] = funcs.loadData(dirPath, optSplit);
    H = optimizeWeightInRange(valiLabels, valiProbs, competingBound, power, padWeight);
    testProbsCorr = funcs.correctProbs(testProbs, H);
    
    rawAcc = funcs.evaluateResult('raw', testLabels, testProbs, 1);
    rawAcc = mean(rawAcc);
    optAcc = funcs.evaluateResult('opt', testLabels, testProbsCorr, 1);
    optAcc = mean(optAcc);
    accuracy_comparison = [rawAcc(4), optAcc(4), optAcc(4)-rawAcc(4)]
    mat
    [rawMaxProb, rawLabels] = max(testProbs, [], 2);
    [optMaxProb, optLabels] = max(testProbsCorr, [], 2);
    
    newTruePositive = find(rawLabels~=optLabels & optLabels==testLabels);
    newFalsePositive = find(rawLabels~=optLabels & rawLabels==testLabels);
    [length(find(rawLabels~=optLabels)) length(newTruePositive), length(newFalsePositive)]
    
end
end

function [probDir, imgDir] = getDirs()
if ~isempty(strfind(pwd, '\CILAB_MACHINE'))
    probDir = 'C:\Users\CILAB_MACHINE\Desktop\CHD\easy-deep-paper\output-data';
else
    probDir = '/home/cideep/Work/tensorflow/output-data';
end

if ~isempty(strfind(pwd, '\CILAB_MACHINE'))
    imgDir = 'C:\Users\CILAB_MACHINE\Desktop\CHD\easy-deep-paper\data\matimg';
else
    imgDir = '/home/cideep/Work/tensorflow/datasets/matimg';
end
end