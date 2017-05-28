function funs = cnnOptFuncs()
  funs.loadData=@loadData;
  funs.optimizeWeight=@optimizeWeight;
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


function H = optimizeWeight(labels, probs)

probs_grt = full(ind2vec(labels'))';
probs_pad = [probs, ones(size(labels))];
'L2 optimization'
H = probs_grt'*pinv(probs_pad');
end
