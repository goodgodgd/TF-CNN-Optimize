clc
clear

list_dir = dir('/home/cideep/Work/tensorflow/output-data/**');
list_dir1 = list_dir;
isdir = [list_dir.isdir];
names = {list_dir.name};
list_dir = list_dir(isdir==1 & startsWith(names, '.')==0);

funcs = cnnOptFuncs();
optRange = [0.2 0.5];

for i=7
    dirPath = [list_dir(i).folder, '/', list_dir(i).name]
    [validLabels, validProbs, testLabels, testProbs] = funcs.loadData(dirPath);
    classAcc_raw = funcs.evaluateResult('raw test', testLabels, testProbs, 1, 1);

    H = optimizeWeightHelpWeakness(validLabels, validProbs, 3, 1, 0.1);
    H_diag = diag(H);
    testProbs_corr = funcs.correctProbs(testProbs, H);
    classAcc_cor = funcs.evaluateResult('corrected test', testLabels, testProbs_corr, 1, 2);
    classAcc_cmp = [classAcc_raw(:,[1 3 4]), classAcc_cor(:,4), ...
        classAcc_cor(:,4) - classAcc_raw(:,4)];
    sprintf('%6d  %6d  %6.3f  %6.3f  %6.3f\n', classAcc_cmp')
end
