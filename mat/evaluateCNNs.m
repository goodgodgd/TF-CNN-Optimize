function [meanResult, verbResult] = evaluateCNNs(competingBound, power, padWeight, toClear)
if nargin<4 || (nargin==4 && toClear==1)
    clc
    clear
end
if nargin<1
    competingBound = [0.1 0.8];
    power = 1.5;
    padWeight = 3;
end

splitInd = 2;

% results = evaluateAccuracies(splitInd);
[meanResult, verbResult] = evaluateAccuracies(splitInd, competingBound, power, padWeight);
low_acc_improve_per_dataset = ...
    [mean(meanResult(1:4,9)) mean(meanResult(5:8,9)) mean(meanResult(9:12,9))]
end

function [meanResult, verbResult] = evaluateAccuracies(splitInd, competingBound, power, padWeight)

if ~isempty(strfind(pwd, '\CILAB_MACHINE'))
    datadir = 'C:\Users\CILAB_MACHINE\Desktop\CHD\easy-deep-paper\output-data';
else
    datadir = '/home/cideep/Work/tensorflow/output-data';
end
splitNames = {'train', 'validation', 'test'};
networks = {'inception_resnet_v2', 'inception_v4', 'resnet_v2_50', 'resnet_v2_101'};
datasets = {'cifar10', 'cifar100', 'voc2012'};
indexComb = combvec(1:4, 1:3)';
numCnns = length(indexComb);
funcs = utilFuncs();
meanResult = zeros(numCnns,11);
verbResult = zeros(0,5);

for i=1:numCnns
    netind = indexComb(i,1);
    datind = indexComb(i,2);
    dirPath = [datadir, '/', cell2mat(networks(netind)), '_', cell2mat(datasets(datind))]
    [testLabels, testProbs] = funcs.loadData(dirPath, 'test');
    classAcc_raw = funcs.evaluateResult('raw test', testLabels, testProbs, 1, 0);
    [weightTrainLabels, weightTrainProbs] = funcs.loadData(dirPath, cell2mat(splitNames(splitInd)));
    
    if nargin==1
        H = optimizeWeightBasic(weightTrainLabels, weightTrainProbs);
        lowAccInds = 1:length(testLabels);
        testProbsCorr = funcs.correctProbs(testProbs, H);
    else
        H = optimizeWeightInRange(weightTrainLabels, weightTrainProbs, competingBound, power, padWeight);
        lowAccInds = findLowAccClassSamples(classAcc_raw(:,4), testLabels);
        testProbsCorr = funcs.correctProbsSelected(testProbs, H, lowAccInds);
    end
    
    [augmAccuracy, classAcc_cor] = funcs.evaluateResultSeperate('corrected test', ...
        testLabels, testProbs, testProbsCorr, lowAccInds, 0, 0);
    
    meanResult(i,:) = summarizeResult(testLabels, testProbs, testProbsCorr, augmAccuracy, ...
        classAcc_raw, classAcc_cor);
    verbResult = [verbResult; concatResult(datind, netind, classAcc_raw, classAcc_cor)];
end
end


function lowAccInds = findLowAccClassSamples(classAccuracy, labels)
numClasses = max(labels);
classAccuracy = [(1:numClasses)', classAccuracy];
classAccuracy = sortrows(classAccuracy, 2);
numLowClasses = round(numClasses*0.2);
lowClassInds = classAccuracy(1:numLowClasses, :);
lowAccInds = [];
for i=1:size(lowClassInds,1)
    lowAccInds = [lowAccInds; find(labels==lowClassInds(i))];
end
end


function summary = summarizeResult(testLabels, testProbs, testProbsCorr, augmAccuracy, ...
                                    classAcc_raw, classAcc_cor)
funcs = utilFuncs();
errorBef = funcs.L2Error(testLabels, testProbs) / 1000;
errorAft = funcs.L2Error(testLabels, testProbsCorr) / 1000;

classAcc_cmp = [classAcc_raw(:,4), classAcc_cor(:,4), classAcc_cor(:,4) - classAcc_raw(:,4)];
classAcc_mean = mean(classAcc_cmp);
% classAcc_mean = [mean accuracy of classes, mean accuracy of corrected classes, diff]
% augmAccuracy = [total accuracy before, total accuracy corrected, diff
%           low class accuracy before, low class accuracy corrected, diff]
summary = [classAcc_mean, augmAccuracy, errorBef, errorAft];
size(summary)
end


function output = concatResult(datind, netind, accRaw, accCor)
numClass = size(accRaw,1);
output = [ones(numClass,1)*datind, ones(numClass,1)*netind, accRaw(:,[1 4]), accCor(:,4)];
end
