clc
clear

dirname = '/home/cideep/Work/tensorflow/output-data/inception_resnet_v2_cifar100';
% dirname = '/home/cideep/Work/tensorflow/output-data/inception_v4_cifar100';
precision = 1e-03;
data = load([dirname, '/test.mat']);
data = data.data;
data_sum = sum(data,2);
valid_inds = find(data_sum>0.1);
data = data(valid_inds,:);
datalen = size(data,1);

labels = data(:,1)+1;
probs = data(:,2:end);
probs_pad = [probs, ones(datalen,1)];
probs_grt = full(ind2vec(labels'))';

numClass = max(labels);
[sortedData] = sortrows(data, 1);
sortedData(abs(sortedData)<precision) = 0;


[correct, total, accuracy] = evaluateAccuracy(labels, probs);
totalAccuracyResult = [correct, total, accuracy];
sprintf('raw total accuracy: %d  %d  %.3f\n', totalAccuracyResult)

classAccuracyResult = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs(inds,:));
    classAccuracyResult(i,:) = [i, correct, total, accuracy];
end
sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')
figure(1)
bins = 0:0.05:1;
histogram(classAccuracyResult(:,4), bins)


%% optimize H by probabilities of within range 0.2~0.5
opt_range = [0.1 0.5];
grt_inds = sub2ind(size(probs), (1:datalen)', labels);
grt_probs = probs(grt_inds);

range_inds = find(grt_probs >= opt_range(1) & grt_probs <= opt_range(2));
numRange = length(range_inds);
labels_range = labels(range_inds);
probs_range = probs(range_inds,:);
probs_grt_range = probs_grt(range_inds,:);
probs_grt_range_rowsum = sum(probs_grt_range);
% probs_grt_range_colsum = sum(probs_grt_range,2);

[correct, total, accuracy] = evaluateAccuracy(labels_range, probs_range);
totalAccuracyResult = [correct, total, accuracy];
sprintf('range accuracy: %d  %d  %.3f\n', totalAccuracyResult)

probs_range_pad = [probs_range, ones(numRange,1)];
H_range = probs_grt_range'*pseudoInverse(probs_range_pad');
% H_range_div = H_range/1000;

mix_ratio = 0.1;
H_range_mix = H_range*mix_ratio + eye(size(H_range))*(1-mix_ratio);
probs_opt_range = probs_range_pad * H_range_mix';

[correct, total, accuracy] = evaluateAccuracy(labels_range, probs_opt_range);
totalAccuracyResult = [correct, total, accuracy];
sprintf('range opt accuracy: %d  %d  %.3f\n', totalAccuracyResult)

probs_opt_range_total = probs_pad * H_range_mix';

[correct, total, accuracy] = evaluateAccuracy(labels, probs_opt_range_total);
totalAccuracyResult = [correct, total, accuracy];
sprintf('range opt total accuracy: %d  %d  %.3f\n', totalAccuracyResult)

classAccuracyResult = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs_opt_range_total(inds,:));
    classAccuracyResult(i,:) = [i, correct, total, accuracy];
end
sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')
figure(2)
bins = 0:0.05:1;
histogram(classAccuracyResult(:,4), bins)

zero_inds = find(classAccuracyResult(:,2)==0);
if ~isempty(zero_inds)
    H_zinds = sub2ind(size(H_range_mix), zero_inds, zero_inds);
    H_range_mix(H_zinds)
    H_diag = diag(H_range_mix);
    [hval, hind] = sort(H_diag);
    [hval(1:10), hind(1:10), probs_grt_range_rowsum(hind(1:10))']
end

return


%% L1 L2 optimization
optres = load([dirname, '/optres_test.mat']);
H_L1 = optres.H_L1;
H_L1_filter = H_L1;
H_L1_filter(abs(H_L1)<precision) = 0;

H_L2 = optres.H_L2;
H_L2_filter = H_L2;
H_L2_filter(abs(H_L2)<precision) = 0;

probs_opt_L1 = probs_pad * H_L1;
probs_opt_L2 = (H_L2 * probs_pad')';

[correct, total, accuracy] = evaluateAccuracy(labels, probs_opt_L1);
totalAccuracyResult = [correct, total, accuracy];
sprintf('optL1 total accuracy: %d  %d  %.3f\n', totalAccuracyResult)

classAccuracyResult = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs_opt_L1(inds,:));
    classAccuracyResult(i,:) = [i, correct, total, accuracy];
end
sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')

%% draw histogram of 'true' probability
selectedProbs = -ones(datalen,1);
for i=1:numClass
    if classAccuracyResult(i,4) > 0.7
        continue
    end
    inds = find(labels==i);
    selectedProbs(inds) = probs(inds,i);
end
selectedProbs = selectedProbs(selectedProbs>-0.1);
histogram(selectedProbs, 20)


%% manually adjust mean of probabilities of each class to be 1
classMean = zeros(numClass,1);
for i=1:numClass
    inds = find(labels==i);
    classMean(i) = mean(probs(inds,i));
end
classWeight = ones(numClass,1)./classMean;
classBias = 1 - classMean;
[(1:numClass)', classMean, classWeight, classBias];
% H_mean = diag(classWeight);
% probs_opt_mean = probs * H_mean;
H_mean = [eye(numClass); classBias'];
probs_opt_mean = probs_pad * H_mean;

[correct, total, accuracy] = evaluateAccuracy(labels, probs_opt_mean);
totalAccuracyResult = [correct, total, accuracy];
sprintf('manual total accuracy: %d  %d  %.3f\n', totalAccuracyResult)

classAccuracyResult = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs_opt_mean(inds,:));
    classAccuracyResult(i,:) = [i, correct, total, accuracy];
end
sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')


%% adjust probatility range into 0.5~1 and optimize H
probs_exp = 0.5 + probs*0.5;
probs_grt_exp = 0.5 + probs_grt*0.5;
probs_exp_pad = [probs_exp, ones(datalen,1)];
H_exp = probs_grt_exp'*pinv(probs_exp_pad');
probs_opt_exp = probs_exp_pad * H_exp';

[correct, total, accuracy] = evaluateAccuracy(labels, probs_opt_exp);
totalAccuracyResult = [correct, total, accuracy];
sprintf('expanded total accuracy: %d  %d  %.3f\n', totalAccuracyResult)

classAccuracyResult = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs_opt_exp(inds,:));
    classAccuracyResult(i,:) = [i, correct, total, accuracy];
end
sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')
