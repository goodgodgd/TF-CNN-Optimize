clc
clear

noc = 10; % number of classes
nod = 1000; % number of data

label = randi(noc,1,nod);
onehot = full(sparse(1:numel(label), label,1));
inpdata = onehot + rand(nod, noc) - 0.5;
smxdata = softmax(inpdata')';

inpcov = cov(inpdata);
smxcov = cov(smxdata);

inpcov_mag = [det(inpcov), trace(inpcov)]
smxcov_mag = [det(smxcov), trace(smxcov)]

figure(1)
subplot(121)
edges = -1:0.02:2;
histogram(inpdata, edges)
axis([-1 2 0 3000])

subplot(122)
edges = 0:0.02:1;
histogram(smxdata, edges)
axis([0 0.5 0 3000])

ce = zeros(nod,1);
for di=1:nod
    ce(di) = -log(smxdata(di,label(di)));
end

crossent = mean(ce)