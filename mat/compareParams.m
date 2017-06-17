clc
clear

lowbnd = [0.05 0.1 0.15];
highbnd = 0.8;
power = 1.5;
padWeight = 3;

resLowBnd = zeros(3,6);
for i=1:3
    competingBound = [lowbnd(i) highbnd];
    result = evaluateCNNs(competingBound, power, padWeight);
    resLowBnd(i,:) = mean(result(:,4:9));
end

figure(1)
subplot(221)
plot(lowbnd, resLowBnd(:,5)*100, 'k-')
xlabel('R low')
axis([lowbnd(1) lowbnd(3) 70 71])

%==================================================
lowbnd = 0.1;
highbnd = [0.7 0.8 0.9];
power = 1.5;
padWeight = 3;

resHighBnd = zeros(3,6);
for i=1:3
    competingBound = [lowbnd highbnd(i)];
    result = evaluateCNNs(competingBound, power, padWeight);
    resHighBnd(i,:) = mean(result(:,4:9));
end

subplot(222)
plot(highbnd, resHighBnd(:,5)*100, 'k-')
xlabel('R high')
axis([highbnd(1) highbnd(3) 70 71])

%==================================================
lowbnd = 0.1;
highbnd = 0.8;
power = [1 1.5 2];
padWeight = 3;

resPower = zeros(3,6);
for i=1:3
    competingBound = [lowbnd highbnd];
    result = evaluateCNNs(competingBound, power(i), padWeight);
    resPower(i,:) = mean(result(:,4:9));
end

subplot(223)
plot(power, resPower(:,5)*100, 'k-')
xlabel('exponent')
axis([power(1) power(3) 70 71])

%==================================================
lowbnd = 0.1;
highbnd = 0.8;
power = 1.5;
padWeight = [2 3 4];

resPad = zeros(3,6);
for i=1:3
    competingBound = [lowbnd highbnd];
    result = evaluateCNNs(competingBound, power, padWeight(i));
    resPad(i,:) = mean(result(:,4:9));
end

subplot(224)
plot(padWeight, resPad(:,5)*100, 'k-')
xlabel('stabil.')
axis([padWeight(1) padWeight(3) 70 71])






