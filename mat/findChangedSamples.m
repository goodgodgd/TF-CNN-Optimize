function findChangedSamples()
clc
clear

competingBound = [0.1 0.8];
power = 1.5;
padWeight = 3;

[probdir, datadir] = getDirs();
optSplit = 'validation';
networks = {'inception_resnet_v2', 'inception_v4', 'resnet_v2_50', 'resnet_v2_101'};
netname = cell2mat(networks(4));
datasets = {'cifar10', 'cifar100', 'voc2012'};
funcs = utilFuncs();

gridCols = 5;
gridRows = [1 1; 4 4; 2 2];
gridRowSum = sum(gridRows);
imageSize = 150;
tpimage = zeros([0 gridCols*imageSize 3], 'uint8');
fpimage = zeros([0 gridCols*imageSize 3], 'uint8');

for datind=1:length(datasets)
    dataname = cell2mat(datasets(datind));
    probDataDir = [probdir, '/', netname, '_', dataname];
    [testLabels, testProbs] = getTestData(probDataDir);
    
    [valiLabels, valiProbs] = funcs.loadData(probDataDir, optSplit);
    H = optimizeWeightInRange(valiLabels, valiProbs, competingBound, power, padWeight);
    testProbsCorr = funcs.correctProbsLowMaxProb(testProbs, H, competingBound(2));
    
    [rawMaxProb, rawLabels] = max(testProbs, [], 2);
    [optMaxProb, optLabels] = max(testProbsCorr, [], 2);
    
    truePosit = find(rawLabels~=optLabels & optLabels==testLabels);
    falsPosit = find(rawLabels~=optLabels & rawLabels==testLabels);
    sampleSize = [length(find(rawLabels~=optLabels)) length(truePosit), length(falsPosit)]

    labelnames = funcs.loadLabelnames([datadir, '/labelname'], dataname)
    tpBefAnnots = createAnnots(truePosit, rawMaxProb, rawLabels, labelnames);
    tpAtfAnnots = createAnnots(truePosit, optMaxProb, optLabels, labelnames);
    fpBefAnnots = createAnnots(falsPosit, rawMaxProb, rawLabels, labelnames);
    fpAtfAnnots = createAnnots(falsPosit, optMaxProb, optLabels, labelnames);

    [images, ~] = funcs.loadImages([datadir, '/matimg'], dataname);
    newtpimage = imagesOnGrid(images, truePosit, [gridRows(datind,1), gridCols], imageSize, ...
        tpBefAnnots, tpAtfAnnots);
    tpimage = [tpimage; newtpimage];
    newfpimage = imagesOnGrid(images, falsPosit, [gridRows(datind,2), gridCols], imageSize, ...
        fpBefAnnots, fpAtfAnnots);
    fpimage = [fpimage; newfpimage];
end
figure(1)
imshow(tpimage)
figure(2)
imshow(fpimage)

imwrite(tpimage, '../../figures/truepos.jpg')
imwrite(fpimage, '../../figures/falspos.jpg')
end

function [probdir, datadir] = getDirs()
probdir = '../../output-data';
if exist('../../data', 'dir')
    datadir = '../../data';
elseif exist('../../datasets', 'dir')
    datadir = '../../datasets';
end
end

function [labels, probs] = getTestData(dirPath)
funcs = utilFuncs();
numSample = 5000;

[labels, probs] = funcs.loadData(dirPath, 'test');
labels = labels(1:numSample);
probs = probs(1:numSample,:);

rndind = randperm(numSample);
% labels = labels(rndind);
% probs = probs(rndind,:);
end


function annots = createAnnots(indices, probs, labels, labelnames)
annots = {};
if ~isrow(indices)
    indices = indices';
end
for idx=indices
    str = sprintf('%s:%.2f', cell2mat(labelnames(labels(idx))), probs(idx));
    annots = [annots, {str}];
end
end


function outimage = imagesOnGrid(images, imginds, gridShape, imageSize, befAnnots, aftAnnots)
rowscols = combvec(1:gridShape(2), 1:gridShape(1))';
indlen = length(rowscols);
assert(indlen < length(imginds));

testrows = 40;
imcols = imageSize;
imrows = imageSize + testrows;
outimage = zeros([gridShape(1)*imrows, gridShape(2)*imcols, 3], 'uint8');

for i=1:indlen
    imgidx = imginds(i);
    row = rowscols(i,2);
    col = rowscols(i,1);
    range = [(row-1)*imrows+1, row*imrows, (col-1)*imcols+1, col*imcols];
    
    curimg = images(:,:,:,imgidx);
    curimg = [imresize(curimg, [imageSize, imageSize]); ones([testrows,imcols,3], 'uint8')*uint8(255)];
    
    textpos = [0, imageSize-5; 0, imageSize+15];
    textstr = {cell2mat(befAnnots(i)), ['>', cell2mat(aftAnnots(i))]};
    curimg = insertText(curimg, textpos, textstr, 'FontSize', 18, 'BoxOpacity', 0);
    
%     imshow(curimg)
%     waitforbuttonpress

    outimage(range(1):range(2), range(3):range(4), :) = curimg;
end
end


function temptest(images, probs, labels)

for i=1:100:2000
    label = labels(i,:)
    prob = probs(i,:)
    img = imresize(images(:,:,:,i), [200,200]);
    imshow(img)
    waitforbuttonpress
end
end
