trainFileName = [dirname, '/', splitName, '.mat']
data = load(trainFileName);
data = data.data;
datalen = size(data,1);
labels = data(:,1)+1;
probs = data(:,2:end);


function optimizeWeight(labels, probs)

probs_grt = full(ind2vec(labels'))';
probs_pad = [probs, ones(datalen,1)];

[probs_opt, H] = L2_optim(probs_grt, probs_pad);
end


function [probs_opt, H] = L2_optim(probs_grt, probs_pad)

'L2 optimization'
H = probs_grt'*pinv(probs_pad');
probs_opt = probs_pad*H';
end
