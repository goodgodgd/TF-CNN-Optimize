function optimizeWeight(dirname, splitName)

trainFileName = [dirname, '/', splitName, '.mat']
data = load(trainFileName);
data = data.data;
datalen = size(data,1);
labels = data(:,1)+1;
probs = data(:,2:end);
% artificial error
% probs(:,1) = min(probs(:,1)+0.3, 1);

probs_grt = full(ind2vec(labels'))';
probs_pad = [probs, ones(datalen,1)];

[probs_opt_L2, H_L2] = L2_optim(probs_grt, probs_pad);
computeErrors(probs_grt, probs_opt_L2, probs)

[probs_opt_L1, H_L1] = L1_optim(probs_grt, probs_pad);
computeErrors(probs_grt, probs_opt_L1, probs)

outputFileName = [dirname, '/optres_', splitName, '.mat']
save(outputFileName, 'H_L2', 'H_L1')
end


function [probs_opt, H] = L2_optim(probs_grt, probs_pad)

'L2 optimization'
H = probs_grt'*pinv(probs_pad');
probs_opt = probs_pad*H';
end


function [probs_opt, H] = L1_optim(probs_grt, probs_pad)

'L1 optimization'
datalen = size(probs_grt,1);
numclass = size(probs_grt,2);
H = zeros(numclass+1, numclass);

for i=numclass:-1:1
    PROBLEM.f = [zeros(numclass+1,1); ones(datalen,1)];
    PROBLEM.Aineq = [probs_pad, -eye(datalen); -probs_pad, -eye(datalen)];
    PROBLEM.bineq = [probs_grt(:,i); -probs_grt(:,i)];
    PROBLEM.lb = zeros(size(PROBLEM.f));
    PROBLEM.solver = 'linprog';
%     PROBLEM.options = optimoptions('linprog', 'Algorithm', 'dual-simplex', ...
%         'MaxIterations', 1000, 'Display', 'off');
    PROBLEM.options = optimoptions('linprog', 'Algorithm', 'interior-point', ...
        'MaxIterations', 1000, 'Display', 'off');
    h = linprog(PROBLEM);
    [i size(h)]
    H(:,i) = h(1:numclass+1)';
end
probs_opt = probs_pad*H;
end


function computeErrors(probs_grt, probs_opt, probs_src)

diff_raw = probs_grt - probs_src;
norm_raw = norm(diff_raw,'fro')
diff_sum_raw = sum(sum(abs(diff_raw)))
diff_opt = probs_grt - probs_opt;
norm_opt = norm(diff_opt,'fro')
diff_sum_opt = sum(sum(abs(diff_opt)))
end
