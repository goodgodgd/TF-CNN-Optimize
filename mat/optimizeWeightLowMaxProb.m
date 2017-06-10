function H = optimizeWeightLowMaxProb(labels, probs, competingBound, power)
datalen = length(labels);
numClasses = size(probs,2);
probGTInds = sub2ind(size(probs), (1:datalen)', labels);
probsAtGT = probs(probGTInds);
probsMax = max(probs,[],2);
competingInds = find(probsAtGT>competingBound(1) & probsMax<competingBound(2));

% create target probatility
probsCompeting = probs(competingInds,:);
probsTarget = probsCompeting;
labelsCompeting = labels(competingInds);
for i=1:length(labelsCompeting)
%     probsTarget(i,:) = createGTProbWithCompetent(probsCompeting(i,:), labelsCompeting(i));
    probsTarget(i,:) = createGTProbByOneHot(probsCompeting(i,:), labelsCompeting(i));
end

% set weight for samples
weightByAccuracy = calcClassWeightByAccuracy(labels, probs, power, competingInds);
% create identical pad
padWeight = calcPadWeight(labels(competingInds), numClasses);
identPad = diag(padWeight);
identPad = identPad(padWeight>0.001,:);
identPad = eye(numClasses)*3;

% set weighted probs
probsTarget = repmat(weightByAccuracy,1,numClasses) .* probsTarget;
probsPad = [probsCompeting, ones(size(labelsCompeting))];
probsPad = repmat(weightByAccuracy,1,numClasses+1) .* probsPad;
% append identical pad
probsTarget = [probsTarget; identPad];
probsPad = [probsPad; [identPad ones(size(identPad(:,1)))]];
% compute wieght
H = probsTarget'*pseudoInverse(probsPad');
a=1;
end


function probGT = createGTProbWithCompetent(probPredict, labelGT)
probOthers = probPredict;
probOthers(labelGT) = 0;
[~, labelMax] = max(probOthers, [], 2);
assert(labelMax~=labelGT, 'labelMax==labelGT')

probGT = probPredict;
sum = probGT(labelGT) + probGT(labelMax);
splitRatio = 0.9;
probGT(labelGT) = sum*splitRatio;
probGT(labelMax) = sum*(1-splitRatio);
probDiff = probGT - probPredict;
end


function probGT = createGTProbByOneHot(probPredict, labelGT)
probGT = zeros(size(probPredict));
probGT(labelGT) = 1;
end


function padWeight = calcPadWeight(labels, numClasses)
edges = 0.5:1:numClasses+0.5;
numSamplesEachClass = histcounts(labels, edges);
maxVal = max(numSamplesEachClass);
padWeight = maxVal - numSamplesEachClass;
end


function wegiht = calcClassWeightByAccuracy(labels, probs, power, competingInds)
funcs = cnnOptFuncs();
classAccuracy = funcs.evaluateResult('training', labels, probs, 0, 0);
classAccuracy = classAccuracy(:,4);
sampleAccuracy = classAccuracy(labels(competingInds));
wegiht = 1./(sampleAccuracy.^power);
end


function printCompetingProbs(probs, labels, changeInds)
[~, maxCols] = max(probs, [], 2);
probsChange = probs(changeInds,:);
changeLen = length(changeInds);

labelChg = labels(changeInds);
maxColsChg = maxCols(changeInds);
labelChgInds = sub2ind([changeLen size(probs,2)], (1:changeLen)', labelChg);
maxColsChgInds = sub2ind([changeLen size(probs,2)], (1:changeLen)', maxColsChg);
changedProbInds = [probsChange(labelChgInds), ...
                    probsChange(maxColsChgInds), labelChg, maxColsChg];
changedProbInds = sortrows(changedProbInds, [3 4]);
changedProbInds = [(1:length(changedProbInds))', changedProbInds];
% sprintf('label prob, max prob, label ind, max ind')
% sprintf('%5d %8.3f %8.3f %5d %5d\n', changedProbInds')

% edges = (0:max(labelChg))+0.1;
% [counts, edges] = histcounts(labelChg, edges);
% [counts', edges(1:length(counts))'+1]
end
