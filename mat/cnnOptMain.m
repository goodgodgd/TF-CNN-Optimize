clc
clear

list_dir = dir('/home/cideep/Work/tensorflow/output-data/**');
isdir = [list_dir.isdir];
names = {list_dir.name};
list_dir = list_dir(isdir==1 & startsWith(names, '.')==0);

funcs = cnnOptFuncs();
optRange = [0.2 0.5];

for i=4
    dirPath = [list_dir(i).folder, '/', list_dir(i).name]
    [validLabels, validProbs, testLabels, testProbs] = funcs.loadData(dirPath);
    funcs.evaluateResult('raw test', testLabels, testProbs, 0, 1);

    H = funcs.optimizeWeightWithWeight(validLabels, validProbs, 0.5, 0.1);
    H_diag = diag(H);
    testProbs_corr = funcs.correctProbs(testProbs, H);
    funcs.evaluateResult('corrected test', testLabels, testProbs_corr, 0, 2);
end
