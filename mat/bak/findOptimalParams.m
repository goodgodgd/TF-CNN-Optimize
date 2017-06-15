clc
clear

performfile = '/home/cideep/Work/tensorflow/output-data/optres.mat';
data = load(performfile);
data = data.result;
data(:,11) = data(:,10) - data(:,9);

accdiff_row = 11;
datatrain = sortrows(data(data(:,4)==1,:),accdiff_row);
datavalid = sortrows(data(data(:,4)==2,:),accdiff_row);
datatest =  sortrows(data(data(:,4)==3,:),accdiff_row);
