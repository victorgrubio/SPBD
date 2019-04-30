% Author: Victor Garcia Rubio
% Homework 2 of Signal Processing for Big Data

function c = ttt(varargin)

%TTT Tensor mulitplication (tensor times tensor).
% 
%   TTT(X,Y) computes the outer product of tensors X and Y.
%
%   TTT(X,Y,XDIMS,YDIMS) computes the contracted product of tensors 
%   X and Y in the dimensions specified by the row vectors XDIMS and 
%   YDIMS.  The sizes of the dimensions specified by XDIMS and YDIMS 
%   must match; that is, size(X,XDIMS) must equal size(Y,YDIMS). 
%
%   TTT(X,Y,DIMS) computes the inner product of tensors X and Y in the
%   dimensions specified by the vector DIMS.  The sizes of the
%   dimensions specified by DIMS must match; that is, size(X,DIMS) must
%   equal size(Y,DIMS). 
%
%   Examples
%   X = tensor(rand(4,2,3));
%   Y = tensor(rand(3,4,2));
%   Z = ttt(X,Y) %<-- outer product of X and Y
%   Z = ttt(X,X,1:3) %<-- inner product of X with itself
%   Z = ttt(X,Y,[1 2 3],[2 3 1]) %<-- inner product of X & Y
%   Z = ttt(X,Y,[1 3],[2 1]) %<-- product of X & Y along specified dims
%
%   See also TENSOR, TENSOR/TTM, TENSOR/TTV.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt



%%%%%%%%%%%%%%%%%%%%%%
%%% ERROR CHECKING %%%
%%%%%%%%%%%%%%%%%%%%%%

% Check the number of arguments
if (nargin < 2)
    error('TTT requires at least two arguments.');
end

% Check the if first two arguments are tensors
if ~isa(varargin{1}, 'tensor') || ~isa(varargin{2}, 'tensor')
    error('First two arguments must be tensors.');
else
    t1 = varargin{1};
    t2 = varargin{2};
end

% Optional 3rd argument
if nargin >= 3
    t1dims = varargin{3};
else
    t1dims = [];
end

% Optional 4th argument
if nargin >= 4
    t2dims = varargin{4};
else
    t2dims = t1dims;
end

if ~isequal(size(t1,t1dims),size(t2,t2dims))
    error('Specified dimensions do not match.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% COMPUTE THE PRODUCT %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%First tensor as matrix
t1mat = ten2mat(t1,t1dims,'t');
%Second tensor
t2mat = ten2mat(t2,t2dims);
%Product
final_mat = t1mat * t2mat;

% Check for scalars
if ~isa(final_mat,'tenmat')
    c = final_mat;
else
    c = tensor(final_mat);
end
