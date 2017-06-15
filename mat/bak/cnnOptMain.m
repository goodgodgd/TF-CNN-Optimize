clc
clear

list_dir = dir('/home/cideep/Work/tensorflow/output-data/**');
isdir = [list_dir.isdir];
names = {list_dir.name};
list_dir = list_dir(isdir==1 & startsWith(names, '.')==0);
names = {list_dir.name}';

funcs = cnnOptFuncs();
weightOptSplit = 'validation';
competingBound = [0 0.6];
power = 0;

for i=1:length(names)
    dirPath = [list_dir(i).folder, '/', list_dir(i).name]
    [testLabels, testProbs] = funcs.loadData(dirPath, 'test');
    classAcc_raw = funcs.evaluateResult('raw test', testLabels, testProbs, 0, 1);

    [weightTrainLabels, weightTrainProbs] = funcs.loadData(dirPath, weightOptSplit);
    H = optimizeWeightLowMaxProb(weightTrainLabels, weightTrainProbs, competingBound, power);
    H_diag = diag(H);

    [augmAccuracy, classAcc_cor] = funcs.evaluateResultSeperate('corrected test', ...
        testLabels, testProbs, H, competingBound, 0, 2);
    classAcc_cmp = [classAcc_raw(:,[1 4]), classAcc_cor(:,4), ...
        classAcc_cor(:,4) - classAcc_raw(:,4)];
%     sprintf('%6d  %6d  %6.3f  %6.3f\n', classAcc_cmp')
    classAcc_mean = mean(classAcc_cmp(:,2:4))
    sprintf('%6d  %6d  %6.3f\n', augmAccuracy')
    names(i)
    waitforbuttonpress
end

comp = [H_diag classAcc_raw(1:end-1,4)];
