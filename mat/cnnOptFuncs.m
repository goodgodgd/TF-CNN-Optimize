function funs = cnnOptFuncs()
  funs.loadData=@loadData;
  funs.optimizeWeightWithWeight=@optimizeWeightWithWeight;
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


function probsCorrected = correctProbs(probs, H)
probsPad = [probs, ones(size(probs,1),1)];
probsCorrected = probsPad*H';
end


function classAccuracyResult = evaluateResult(scopeStr, labels, probs, evalPerClass, histFigNum)
[correct, total, accuracy] = evaluateAccuracy(labels, probs);
totalAccuracyResult = [correct, total, accuracy];
sprintf('%s, total accuracy: %d  %d  %.3f\n', scopeStr, totalAccuracyResult)

numClass = size(probs,2);
if nargin>3
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

if nargin>4 && histFigNum>0
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

