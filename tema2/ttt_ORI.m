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

%%%%%%%%%%%%%%%%%%%%%%
%%% ERROR CHECKING %%%
%%%%%%%%%%%%%%%%%%%%%%

% Check the number of arguments
if (nargin < 2)
    error('TTT requires at least two arguments.');
end
a = varargin{1};
b = varargin{2};

% Optional 3rd argument 
if nargin >= 3
    adims = varargin{3};
else  %<--- here is the outer product
    adims = [];
end

% Optional 4th argument
if nargin >= 4
    bdims = varargin{4};
else
    bdims = adims;
end

if ~isequal(size(a,adims),size(b,bdims))
    error('Specified dimensions do not match.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% COMPUTE THE PRODUCT %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% computing C = A * B
$$$$$$ it might require here some if statement to differentiate outer product
$$$$$$ from inner products. Perhaps a squeeze call might also be needed 
amatrix = tenmat(a,adims,'t'); $$$$$$ change here tenmat -> ten2mat
                               $$$$$$ inputs arguments doesn't have to coincide
bmatrix = tenmat(b,bdims); $$$$$$ change here tenmat -> ten2mat
                           $$$$$$ inputs arguments doesn't have to coincide
cmatrix = amatrix * bmatrix;

% Check whether or not the result is a outer product.
$$$$$$ it might require here some if statement to differentiate outer product
$$$$$$ from inner products 
c = cmatrix;
