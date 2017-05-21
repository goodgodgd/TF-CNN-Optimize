function [correct, total, accuracy] = evaluateAccuracy(labels, probs)

[~, maxInd] = max(probs, [], 2);
correct = sum(maxInd == labels);
total = length(labels);
accuracy = correct/total;
end
