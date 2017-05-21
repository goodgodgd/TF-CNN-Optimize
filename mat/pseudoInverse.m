function M_pinv = pseudoInverse(M, tolorence)

if nargin<2
    tolerance = 0.001;
end

[m, n] = size(M);
[U, S, V] = svd(M);
M_recon = U*S*V';
% svd_error = sum(sum(abs(M - M_recon)))

S_diag = diag(S);
S_norm = S_diag/sum(abs(S_diag));
valid = (S_norm < tolerance);
inval = ~valid;
S_diag(inval) = 1;
S_inv_diag = 1./S_diag;
S_inv_diag(valid) = 0;

if m==n
    S_inv = diag(S_inv_diag);
elseif m<n
    S_inv = [diag(S_inv_diag); zeros(n-m,m)];
elseif m>n
    S_inv = [diag(S_inv_diag), zeros(n,m-n)];
end

M_pinv = V*S_inv*U';
end