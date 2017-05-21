function rnk = normRank(M, tolerance)

if nargin<2
    tolerance = 0.001;
end

[~,S,~] = svd(M);
S = diag(S);
S = S / sum(abs(S));
S = sort(S,1,'descend');
S_top5 = S(1:5)
rnk = length(S(S>tolerance));
end