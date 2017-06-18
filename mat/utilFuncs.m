function funs = utilFuncs()
  funs.loadData=@loadData;
  funs.optimizeWeightWithWeight=@optimizeWeightWithWeight;
  funs.correctProbsSelected=@correctProbsSelected;
  funs.correctProbs=@correctProbs;
  funs.evaluateResult=@evaluateResult;
  funs.evaluateResultSeperate=@evaluateResultSeperate;
  funs.L2Error=@L2Error;
end


function [labels, probs] = loadData(path, split, filter)
if nargin < 3
    filter = 1;
end
fileName = sprintf('%s/%s.mat', path, split);
data = load(fileName);
data = data.data;
labels = data(:,1)+1;
probs = data(:,2:end);
if filter==0
    return
end
probSum = sum(probs,2);
inds = find(probSum<0.9);
minInd = min(inds);
probs = probs(abs(probSum-1)<0.001,:);
labels = labels(abs(probSum-1)<0.001);
end


function probsCorrected = correctProbsSelected(probs, H, selectInds)
probsSelectCorr = correctProbs(probs(selectInds,:), H);
probsCorrected = probs;
probsCorrected(selectInds,:) = probsSelectCorr;
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
    probsCorr, selectInds, printAccuracy, histFigNum)

% total accuracy before and after correction
totalBefAccuracy = evaluateAccuracy(labels, probs);
totalCorrAccuracy = evaluateAccuracy(labels, probsCorr);

% low accuracy class samples before and after correction
selectAccuracyBef = evaluateAccuracy(labels(selectInds), probs(selectInds,:));
selectAccuracyCorr = evaluateAccuracy(labels(selectInds), probsCorr(selectInds,:));

sprintf('%s, total accuracy: %d  %d  %.3f\n', scopeStr, totalCorrAccuracy)
augmAccuracy = [totalBefAccuracy(3) totalCorrAccuracy(3) ...
    totalCorrAccuracy(3) - totalBefAccuracy(3) ...
    selectAccuracyBef(3) selectAccuracyCorr(3) ...
    selectAccuracyCorr(3) - selectAccuracyBef(3)];

numClass = size(probs,2);
classAccuracy = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    classAccuracy(i,:) = [i evaluateAccuracy(labels(inds), probsCorr(inds,:))];
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


function err = L2Error(labels, probs)
datalen = length(labels);
probGTInds = sub2ind(size(probs), (1:datalen)', labels);
probsTarget = zeros(size(probs));
probsTarget(probGTInds) = 1;

probDiff = probs - probsTarget;
err = sum(sum(probDiff.*probDiff));
end
