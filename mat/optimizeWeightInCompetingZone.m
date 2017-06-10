function H = optimizeWeightInCompetingZone(labels, probs, competingZone, restWeight)
datalen = length(labels);
numClasses = size(probs,2);
probGTInds = sub2ind(size(probs), (1:datalen)', labels);
probsAtGT = probs(probGTInds);
competingTF = (probsAtGT>competingZone(1) & probsAtGT<competingZone(2));
competingInds = find(competingTF);

printCompetingProbs(probs, labels, competingTF);

% create target probatility
probsTarget = probs;
for ci = competingInds'
%     probsTarget(ci,:) = createGTProbWithCompetent(probs(ci,:), labels(ci));
    probsTarget(ci,:) = createGTProbByOneHot(probs(ci,:), labels(ci));
end

% set weight for samples
weightForCompetents = ones(datalen,1) * restWeight;
weightForCompetents(competingInds) = 1;
weightByClassSize = calcWeightByClassSize(labels, competingInds);
% classWeightForComepetents = calcClassWeightForComepetents(labels, probs, competingInds);
weightVec = weightForCompetents .* weightByClassSize;
tmp = [labels, competingTF, weightForCompetents, weightByClassSize, weightVec];

probsTarget = repmat(weightVec,1,numClasses) .* probsTarget;
probsPad = [probs, ones(size(labels))];
probsPad = repmat(weightVec,1,numClasses+1) .* probsPad;
H = probsTarget'*pseudoInverse(probsPad');
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
end


function probGT = createGTProbByOneHot(probPredict, labelGT)
probGT = zeros(size(probPredict));
probGT(labelGT) = 1;
end


function weight = calcWeightByClassSize(labels, competingInds)
if nargin < 2
    labelForCount = labels;
else
    labelForCount = labels(competingInds);
end
numClasses = max(labels);
edges = (0:numClasses) + 0.5;
classCounts = histcounts(labelForCount, edges)' + 2;
classWeight = max(classCounts)./classCounts;
weight = classWeight(labels);
end


function printCompetingProbs(probs, labels, changeTF)
[~, maxCols] = max(probs, [], 2);

probsChange = probs(changeTF,:);
changeInds = find(changeTF);
changeLen = length(changeInds);

labelChg = labels(changeTF);
maxColsChg = maxCols(changeTF);
labelChgInds = sub2ind([changeLen size(probs,2)], (1:changeLen)', labelChg);
maxColsChgInds = sub2ind([changeLen size(probs,2)], (1:changeLen)', maxColsChg);
changedProbInds = [probsChange(labelChgInds), ...
                    probsChange(maxColsChgInds), labelChg, maxColsChg];
changedProbInds = sortrows(changedProbInds, [3 4]);
changedProbInds = [(1:length(changedProbInds))', changedProbInds];
sprintf('label prob, max prob, label ind, max ind')
sprintf('%5d %8.3f %8.3f %5d %5d\n', changedProbInds')

% edges = (0:max(labelChg))+0.1;
% [counts, edges] = histcounts(labelChg, edges);
% [counts', edges(1:length(counts))'+1]
end
