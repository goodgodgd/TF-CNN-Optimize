function H = optimizeWeightBasic(labels, probs)

datalen = length(labels);
probGTInds = sub2ind(size(probs), (1:datalen)', labels);
probsTarget = zeros(size(probs));
probsTarget(probGTInds) = 1;
probsPad = [probs, ones(size(labels))];
% compute wieght
H = probsTarget'*pseudoInverse(probsPad');
end
