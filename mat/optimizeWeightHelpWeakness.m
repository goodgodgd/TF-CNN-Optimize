function H = optimizeWeightHelpWeakness(labels, probs, power, restWeight, minGTProb)

datalen = length(labels);
numClasses = max(labels);

[~, maxCols] = max(probs, [], 2);
probGTInds = sub2ind(size(probs), (1:datalen)', labels);
probsAtGT = probs(probGTInds);
probMaxInds = sub2ind(size(probs), (1:datalen)', maxCols);
probsAtMax = probs(probMaxInds);
changeTF = (labels~=maxCols & probsAtGT>minGTProb & probsAtMax<probsAtGT*2);
changeInds = find(changeTF);

printChangingProbs(probs, labels, changeTF);

classWeight = classWeightByAccuracy(labels, probs, power, changeInds);
[(1:numClasses)', classWeight]

probsTarget = probs;
for ci = changeInds'
    probsTarget(ci,:) = createTargetProb(probs(ci,:), labels(ci));
end
weightVec = ones(datalen,1) * restWeight;
weightVec(changeInds) = classWeight(labels(changeInds));

probsTarget = repmat(weightVec,1,numClasses) .* probsTarget;
probsPad = [probs, ones(size(labels))];
probsPad = repmat(weightVec,1,numClasses+1) .* probsPad;
H = probsTarget'*pseudoInverse(probsPad');
end


function classWeight = classWeightByAccuracy(labels, probs, power, changeInds)

numClasses = max(labels);
counts = histcounts(labels(changeInds), 0.5:1:numClasses+0.5) + 2;
normCounts = counts'./max(counts)

funcs = cnnOptFuncs();
classAccuracy = funcs.evaluateResult('help weakness', labels, probs, 0, 0);
classAccuracy = classAccuracy(1:numClasses,4);
classWeight = 1./(classAccuracy.^power)./normCounts;
end


function probGT = createTargetProb(probPred, labelGT)
probGT = probPred;
[~, labelMax] = max(probPred, [], 2);
assert(labelMax~=labelGT, 'labelMax==labelGT')

% probsToChange = [probGT(labelGT) probGT(labelMax)]
sum = probGT(labelGT) + probGT(labelMax);
splitRatio = 0.8;
probGT(labelGT) = sum*splitRatio;
probGT(labelMax) = sum*(1-splitRatio);
end


function printChangingProbs(probs, labels, changeTF)
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
