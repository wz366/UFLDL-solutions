function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

m = size(data, 2);

z2 = W1 * data;
z2 = z2 + repmat(b1, [1 size(z2, 2)]);
a2 = sigmoid(z2);

z3 = W2 * a2;
z3 = z3 + repmat(b2, [1 size(z3, 2)]);
a3 = sigmoid(z3);

errors = a3 - data;
cost = 1/m * 1/2 * sum(sum(errors.^2));
penalty1 = lambda/2 * (sum(sum(W1.^2))+sum(sum(W2.^2)));
p = mean(a2, 2);
KL = sparsityParam .* log(sparsityParam ./ p) + (1-sparsityParam) .* log((1-sparsityParam) ./ (1 - p));
penalty2 = beta * sum(KL);
cost = cost + penalty1 + penalty2;

errorTerm3 = errors .* sigmoidGradient(z3);
errorTerm2 = W2' * errorTerm3;
t = beta * (-sparsityParam ./ p + (1-sparsityParam) ./ (1 - p));
errorTerm2 = errorTerm2 + repmat(t, [1 size(errorTerm2, 2)]);
errorTerm2 = errorTerm2 .* sigmoidGradient(z2);

partialW2 = errorTerm3 * a2';
partialW1 = errorTerm2 * data';
partialb2 = sum(errorTerm3, 2);
partialb1 = sum(errorTerm2, 2);

W1grad = 1/m * partialW1  + lambda * W1;
W2grad = 1/m * partialW2 + lambda * W2; 
b1grad = 1/m * partialb1;
b2grad = 1/m * partialb2;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function g = sigmoidGradient(z)
	a = sigmoid(z);
	g = a .* (1 - a);
end