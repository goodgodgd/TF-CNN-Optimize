function evaluateOptParams()

trySingleCombParams();
% saveResultAllCombParams();

end


function trySingleCombParams()
clc
clear
splitInd = 2;
competingBound = [0.1 0.7];
power = 1.5;
eachResult = evaluateParams(splitInd, competingBound, power)
[mean(eachResult(1:4,9)) mean(eachResult(5:8,9)) mean(eachResult(9:12,9))]
end


function saveResultAllCombParams()
lowBound = 0:0.05:0.3;
uppBound = 0.5:0.05:0.7;
powers = 0:0.5:2;
splitInds = 1:3;


boundComb = combvec(lowBound, uppBound, powers, splitInds)';
numCombs = length(boundComb)
result = zeros(numCombs, 6);

for i=1:numCombs
    comb_index = i
    eachResult = evaluateParams(cell2mat(splitNames(boundComb(i,4))), boundComb(i,1:2), boundComb(i,3));
    eachResult = eachResult(eachResult(:,1) > 0.1, :);
    if size(eachResult,1) > 5
        result(i,:) = [size(eachResult,1) mean(eachResult,1)];
    else
        result(i,:) = zeros(1,6);
    end
end

result = [boundComb result];
result = result(result(:,5) > 0, :);

outputfile = '/home/cideep/Work/tensorflow/output-data/optres.mat';
save(outputfile, 'result');
end


function result = evaluateParams(splitInd, competingBound, power)

splitNames = {'train', 'validation', 'test'};
datadir = '/home/cideep/Work/tensorflow/output-data';
networks = {'inception_resnet_v2', 'inception_v4', 'resnet_v2_50', 'resnet_v2_101'};
datasets = {'cifar10', 'cifar100', 'voc2012'};
indexComb = combvec(1:4, 1:3)';
numCnns = length(indexComb);
funcs = cnnOptFuncs();
result = zeros(numCnns,9);

for i=1:numCnns
    netind = indexComb(i,1);
    datind = indexComb(i,2);
    dirPath = [datadir, '/', cell2mat(networks(netind)), '_', cell2mat(datasets(datind))]
    [testLabels, testProbs] = funcs.loadData(dirPath, 'test');
    classAcc_raw = funcs.evaluateResult('raw test', testLabels, testProbs, 0, 0);

    [weightTrainLabels, weightTrainProbs] = funcs.loadData(dirPath, cell2mat(splitNames(splitInd)));
    H = optimizeWeightLowMaxProb(weightTrainLabels, weightTrainProbs, competingBound, power);
    H_diag = diag(H);
    
    lowAccInds = findLowAccClassSamples(classAcc_raw(:,4), testLabels);
    [augmAccuracy, classAcc_cor] = funcs.evaluateResultSeperate('corrected test', ...
        testLabels, testProbs, H, lowAccInds, 0, 0);
    
    classAcc_cmp = [classAcc_raw(:,4), classAcc_cor(:,4)];
%     sprintf('%6d  %6.3f  %6.3f  %6.3f\n', classAcc_cmp')
%     sprintf('%6d  %6d  %6.3f\n', augmAccuracy')\
    
    % result = [mean accuracy of classes, mean accuracy of corrected classes, diff
    %           total accuracy before, total accuracy corrected, diff
    %           low class accuracy before, low class accuracy corrected, diff
    classAcc_mean = mean(classAcc_cmp);
    classAcc_mean = [classAcc_mean, classAcc_mean(2)-classAcc_mean(1)];
    result(i,:) = [classAcc_mean, augmAccuracy];
%     waitforbuttonpress
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