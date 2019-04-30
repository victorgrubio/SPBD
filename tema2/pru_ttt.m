%% NOTES
% VARARGIN TO CHECK ARGUMENTS AS INPUT
% 

%% CODE

% some code to check function ttt
clear all; close all; clc;
X = rand(4,2,3);
Y = rand(3,4,2);
X = tensor(X);
Y = tensor(Y);
Z1 = ttt(X,Y); %<-- outer product of X and Y
Z2 = ttt(X,X,1:3); %<-- inner product of X with itself
Z3 = ttt(X,Y,[1 2 3],[2 3 1]); %<-- inner product of X & Y
Z4 = ttt(X,Y,[1 3],[2 1]); %<-- product of X & Y along specified dims
%
size(Z1),size(Z2),size(Z3),size(Z4)
% should be:
% 4     2     3     3     4     2  <-- a 6 order tensor
% 1 1 <-- a scalar (or 1x1 vector)
% 1 1 <-- a scalar (or 1x1 vector)
% 2 2 <-- a matrix (or 2 order tensor) 
