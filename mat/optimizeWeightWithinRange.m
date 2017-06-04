function H = optimizeWeightWithinRange(labels, probs, optRange, mix_ratio)
datalen = length(labels);
grtInds = sub2ind(size(probs), (1:datalen)', labels);
grtProbs = probs(grtInds);

rangeInds = find(grtProbs >= optRange(1) & grtProbs <= optRange(2));
labelsRange = labels(rangeInds);
probsRange = probs(rangeInds,:);

H_range = optimizeWeight(labelsRange, probsRange);
H = H_range*mix_ratio + eye(size(H_range))*(1-mix_ratio);
end


function H = optimizeWeight(labels, probs)
probsGT = full(ind2vec(labels'))';
probsPad = [probs, ones(size(labels))];
H = probsGT'*pseudoInverse(probsPad');
end
