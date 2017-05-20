function evaluateResult(dirname)

'--------------------------------------------------'
data = load([dirname, '/test.mat']);
data = data.data;
datalen = size(data,1);
labels = data(:,1)+1;
probs = data(:,2:end);
probs_pad = [probs, ones(datalen,1)];
probs_grt = full(ind2vec(labels'))';

optres = load([dirname, '/optres_test.mat']);
probs_opt = probs_pad * optres.H_L1;

'evaluate accuracy for original test data'
evaluateAccuracyTotal(labels, probs);
evaluateAccuracyEachClass(labels, probs);

'evaluate accuracy for optimized test data'
evaluateAccuracyTotal(labels, probs_opt);
evaluateAccuracyEachClass(labels, probs_opt);

computeErrors(probs_grt, probs_opt, probs)
end


function evaluateAccuracyTotal(labels, probs)

[correct, total, accuracy] = evaluateAccuracy(labels, probs);
totalAccuracyResult = [correct, total, accuracy];
sprintf(['%d  %d  %.3f\n'], totalAccuracyResult)
end


function evaluateAccuracyEachClass(labels, probs)

numClass = max(labels);
data = [labels, probs];
[sortedData] = sortrows(data, 1);
labels = sortedData(:,1);
probs = sortedData(:,2:end);

classAccuracyResult = zeros(numClass, 4);
for i=1:numClass
    inds = find(labels==i);
    [correct, total, accuracy] = evaluateAccuracy(labels(inds), probs(inds,:));
    classAccuracyResult(i,:) = [i, correct, total, accuracy];
end
sprintf(['%6d  %6d  %6d  %6.3f\n'], classAccuracyResult')
end


function [correct, total, accuracy] = evaluateAccuracy(labels, probs)

[~, maxInd] = max(probs, [], 2);
correct = sum(maxInd == labels);
total = length(labels);
accuracy = correct/total;
end


function computeErrors(probs_grt, probs_opt, probs_src)

diff_raw = probs_grt - probs_src;
norm_raw = norm(diff_raw,'fro')
diff_sum_raw = sum(sum(abs(diff_raw)))
diff_opt = probs_grt - probs_opt;
norm_opt = norm(diff_opt,'fro')
diff_sum_opt = sum(sum(abs(diff_opt)))
end
