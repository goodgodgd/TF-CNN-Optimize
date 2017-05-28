clc
clear

list_dir = dir('/home/cideep/Work/tensorflow/output-data/**');
isdir = [list_dir.isdir];
names = {list_dir.name};
list_dir = list_dir(isdir==1 & startsWith(names, '.')==0);

funcs = cnnOptFuncs();
optRange = [0.2 0.5];

for i=2
    dirPath = [list_dir(i).folder, '/', list_dir(i).name]
    [validLabels, validProbs, testLabels, testProbs] = funcs.loadData(dirPath);
    funcs.evaluateResult('raw test', testLabels, testProbs, 0, 1);

    H = funcs.optimizeWeightWithinRange(validLabels, validProbs, optRange, 0.2);
    testProbs_corr = funcs.correctProbs(testProbs, H);
    funcs.evaluateResult('corrected test', testLabels, testProbs_corr, 0, 2);
end
