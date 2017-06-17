clc
clear

if ~isempty(strfind(pwd, '\CILAB_MACHINE'))
    datadir = 'C:\Users\CILAB_MACHINE\Desktop\CHD\easy-deep-paper\output-data';
else
    datadir = '/home/cideep/Work/tensorflow/output-data';
end
network = 'inception_v4';
datasets = {'cifar10', 'cifar100', 'voc2012'};
splits = {'train', 'validation', 'test'};
funcs = cnnOptFuncs();

datasize = zeros(3,3);

for datind=1:3
    dirPath = [datadir, '/', network, '_', cell2mat(datasets(datind))];
    for splitind=1:3
        [labels, probs] = funcs.loadData(dirPath, cell2mat(splits(splitind)), 0);
        datasize(datind, splitind) = length(labels);
    end
end

datasize