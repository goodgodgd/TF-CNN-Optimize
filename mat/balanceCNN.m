clc
clear

list_dir = dir('/home/cideep/Work/tensorflow/output-data/**');
isdir = [list_dir.isdir];
names = {list_dir.name};
list_dir = list_dir(isdir==1 & startsWith(names, '.')==0);

trainSplit = 'test';

for i=2
    dirPath = [list_dir(i).folder, '/', list_dir(i).name]
    [validLabels, validProbs, testLabels, testProbs] = loadData(dirPath);
    H = optimizeWeight(validLabels, validProbs)
    evaluateResult(dirpath)
end
