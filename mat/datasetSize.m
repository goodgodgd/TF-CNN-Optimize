clc
clear

datadir = '/home/cideep/Work/tensorflow/output-data';
network = 'inception_v4';
datasets = {'cifar10', 'cifar100', 'voc2012'};
splits = {'train', 'validation', 'test'};
funcs = cnnOptFuncs();

datasize = zeros(3,3);

for datind=1:3
    dirPath = [datadir, '/', network, '_', cell2mat(datasets(datind))];
    for splitind=1:3
        [labels, probs] = funcs.loadData(dirPath, cell2mat(splits(splitind)));
        datasize(datind, splitind) = length(labels);
    end
end

datasize