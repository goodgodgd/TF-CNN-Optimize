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
% probs_opt = optres.H_L2 * probs_pad';
% probs_opt = probs_opt';

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
sprintf('%d  %d  %.3f\n', totalAccuracyResult)
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
sprintf('%6d  %6d  %6d  %6.3f\n', classAccuracyResult')
end


function computeErrors(probs_grt, probs_opt, probs_src)

diff_raw = probs_grt - probs_src;
raw_L2_norm = norm(diff_raw,'fro')
raw_L1_norm = sum(sum(abs(diff_raw)))
diff_opt = probs_grt - probs_opt;
opt_L2_norm = norm(diff_opt,'fro')
opt_L1_norm = sum(sum(abs(diff_opt)))
end
