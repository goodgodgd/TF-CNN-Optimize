clc
clear

list_mat = dir('/home/cideep/Work/tensorflow/output-data/**/*.mat')

list_dir = dir('/home/cideep/Work/tensorflow/output-data/**')
isdir = [list_dir.isdir];
names = {list_dir.name}
list_dir = list_dir(isdir==1 & startsWith(names, '.')==0)
