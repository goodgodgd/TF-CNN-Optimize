function funs = cnnOptFuncs()
  funs.loadData=@loadData;
  funs.optimizeWeightWithinRange=@optimizeWeightWithinRange;
  funs.optimizeWeightWithWeight=@optimizeWeightWithWeight;
  funs.optimizeWeight=@optimizeWeight;
  funs.correctProbs=@correctProbs;
  funs.evaluateResult=@evaluateResult;
end


function [validLabels, validProbs, testLabels, testProbs] = loadData(path)
fileName = [path, '/validation.mat']
data = load(fileName);
data = data.data;
validLabels = data(:,1)+1;
validProbs = data(:,2:end);

fileName = [path, '/test.mat']
data = load(fileName);
data = data.data;
testLabels = data(:,1)+1;
testProbs = data(:,2:end);
end


function H = optimizeWeightWithWeight(labels, probs, stillWeight, minGTProb)
[~, maxCols] = max(probs, [], 2);
datalen = length(labels);
probGTInds = sub2ind(size(probs), (1:datalen)', labels);
probsAtGT = probs(probGTInds);
probMaxInds = sub2ind(size(probs), (1:datalen)', maxCols);
probsAtMax = probs(probMaxInds);
changeTF = (labels~=maxCols & probsAtGT>minGTProb & probsAtMax<probsAtGT*2);
changeInds = find(changeTF);
probsChange = probs(changeTF,:);
changeLen = length(changeInds)


labelChg = labels(changeTF);
[length(find(labelChg==11)) length(find(labelChg==12)) length(find(labelChg==36)) length(find(labelChg==56))]
maxColsChg = maxCols(changeTF);
labelChgInds = sub2ind([changeLen size(probs,2)], (1:changeLen)', labelChg);
maxColsChgInds = sub2ind([changeLen size(probs,2)], (1:changeLen)', maxColsChg);
changedProbInds = [probsChange(labelChgInds), probsChange(maxColsChgInds), labelChg, maxColsChg];
sprintf('%8.3f %8.3f %5d %5d\n', changedProbInds')

% figure(3)
% targetInds = (labels==12);
% bins = 0:0.05:1;
% histogram(probsAtGT(targetInds), bins)


probsGT = probs;
for ci = changeInds'
    probsGT(ci,:) = createGTProb(probs(ci,:), labels(ci));
end

stillInds = (~changeTF);
probsGT(stillInds,:) = stillWeight * probsGT(stillInds,:);
probsPad = [probs, ones(size(labels))];
probsPad(stillInds,:) = stillWeight * probsPad(stillInds,:);
H = probsGT'*pseudoInverse(probsPad');

% probsChangePad = [probs(changeInds,:), ones(length(changeInds),1)];
% H = probsGT(changeInds,:)' * pseudoInverse(probsChangePad');
end


function probGT = createGTProb(probPred, labelGT)
probGT = probPred;
[~, labelMax] = max(probPred, [], 2);
assert(labelMax~=labelGT, 'labelMax==labelGT')

% probsToChange = [probGT(labelGT) probGT(labelMax)]
sum = probGT(labelGT) + probGT(labelMax);
splitRatio = 0.8;
probGT(labelGT) = sum*splitRatio;
probGT(labelMax) = sum*(1-splitRatio);
end


function H = optimizeWeightWithinRange(labels, probs, optRange, mix_ratio)
datalen = length(labels);
grtInds = sub2ind(size(probs), (1:datalen)', labels);
grtProbs = probs(grtInds);

rangeInds = find(grtProbs >= optRange(1) & grtProbs <= optRange(2));
labelsRange = labels(rangeInds);
probsRange = probs(rangeInds,:);

H_range = optimizeWeight(labelsRange, probsRange);
H = H_range*mix_ratio + eye(size(H_range))*(1-mix_ratio);
end


function H = optimizeWeight(labels, probs)
probsGT = full(ind2vec(labels'))';
probsPad = [probs, ones(size(labels))];
H = probsGT'*pseudoInverse(probsPad');
end


function probsCorrected = correctProbs(probs, H)
probsPad = [probs, ones(size(probs,1),1)];
probsCorrected = probsPad*H';
end


function evaluateResult(scopeStr, labels, probs, evalPerClass, histFigNum)
[correct, total, accuracy] = evaluateAccuracy(labels, probs);
totalAccuracyResult = [correct, total, accuracy];
sprintf('%s, total accuracy: %d  %d  %.3f\n', scopeStr, totalAccuracyResult)

numClass = size(probs,2);
if nargin>2
    classAccuracyResult = zeros(numClass, 4);
    for i=1:numClass
        inds = find(labels==i);
        [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs(inds,:));
        classAccuracyResult(i,:) = [i, correct, total, accuracy];
    end
    if evalPerClass>0
        sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')
    end
end

if nargin>3 && histFigNum>0
    figure(histFigNum)
    bins = 0:0.05:1;
    histogram(classAccuracyResult(:,4), bins)
    axis([0 1 0 20])
end
end


function [correct, total, accuracy] = evaluateAccuracy(labels, probs)
[~, maxInd] = max(probs, [], 2);
correct = sum(maxInd == labels);
total = length(labels);
accuracy = correct/total;
end

