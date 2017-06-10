function funs = cnnOptFuncs()
  funs.loadData=@loadData;
  funs.optimizeWeightWithWeight=@optimizeWeightWithWeight;
  funs.correctProbs=@correctProbs;
  funs.evaluateResult=@evaluateResult;
  funs.evaluateResultSeperate=@evaluateResultSeperate;
end


function [labels, probs] = loadData(path, split)
fileName = sprintf('%s/%s.mat', path, split);
data = load(fileName);
data = data.data;
labels = data(:,1)+1;
probs = data(:,2:end);
probSum = sum(probs,2);
probs = probs(abs(probSum-1)<0.001,:);
labels = labels(abs(probSum-1)<0.001);
end


function probsCorrected = correctProbs(probs, H)
probsPad = [probs, ones(size(probs,1),1)];
probsCorrected = probsPad*H';
end


function classAccuracyResult = evaluateResult(scopeStr, labels, probs, evalPerClass, histFigNum)
totalAccuracyResult = evaluateAccuracy(labels, probs);
sprintf('%s, total accuracy: %d  %d  %.3f\n', scopeStr, totalAccuracyResult)

numClass = size(probs,2);
if nargin>3
    classAccuracyResult = zeros(numClass, 4);
    for i=1:numClass
        inds = find(labels==i);
        classAccuracyResult(i,:) = [i, evaluateAccuracy(labels(inds), probs(inds,:))];
    end
    
    if evalPerClass>0
        sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')
    end
end

if nargin>4 && histFigNum>0
    figure(histFigNum)
    bins = 0:0.05:1;
    histogram(classAccuracyResult(:,4), bins)
    axis([0 1 0 20])
end
end


function result = evaluateAccuracy(labels, probs)
[~, maxInd] = max(probs, [], 2);
correct = sum(maxInd == labels);
total = length(labels);
accuracy = correct/total;
result = [correct, total, accuracy];
end


function [augmAccuracy, classAccuracy] = evaluateResultSeperate(scopeStr, labels, probs, ...
    H, selectInds, printAccuracy, histFigNum)

% total accuracy before
totalBefAccuracy = evaluateAccuracy(labels, probs);

% low accuracy class samples before
probsSelect = probs(selectInds,:);
labelsSelect = labels(selectInds);
selectAccuracyBef = evaluateAccuracy(labelsSelect, probsSelect);
% low accuracy class samples after correction
probsSelectCorr = correctProbs(probsSelect, H);
selectAccuracyCorr = evaluateAccuracy(labelsSelect, probsSelectCorr);

% total smaples partially corrected
probsPartCorr = probs;
probsPartCorr(selectInds,:) = probsSelectCorr;
totalPartCorrAccuracy = evaluateAccuracy(labels, probsPartCorr);

sprintf('%s, total accuracy: %d  %d  %.3f\n', scopeStr, totalPartCorrAccuracy)
augmAccuracy = [totalBefAccuracy(3) totalPartCorrAccuracy(3) ...
    totalPartCorrAccuracy(3) - totalBefAccuracy(3) ...
    selectAccuracyBef(3) selectAccuracyCorr(3) ...
    selectAccuracyCorr(3) - selectAccuracyBef(3)];

numClass = size(probs,2);
classAccuracy = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    classAccuracy(i,:) = [i evaluateAccuracy(labels(inds), probsPartCorr(inds,:))];
end

if nargin>3 && printAccuracy>0
    sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracy')
end

if nargin>4 && histFigNum>0
    figure(histFigNum)
    edges = 0:0.05:1;
    histogram(classAccuracy(:,4), edges)
    axis([0 1 0 20])
end
end
