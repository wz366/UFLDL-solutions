function checkNNGradients(lambda, sparsityParam, beta)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
if ~exist('sparsityParam', 'var') || isempty(sparsityParam)
    sparsityParam = 0;
end
if ~exist('beta', 'var') || isempty(beta)
    beta = 0;
end

input_layer_size = 7;
hidden_layer_size = 3;
m = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(input_layer_size, hidden_layer_size);
b1 = zeros(hidden_layer_size, 1);
b2 = zeros(input_layer_size, 1);
% Reusing debugInitializeWeights to generate data
data  = (debugInitializeWeights(m, input_layer_size))';
% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:) ; b1(:) ; b2(:)];

% Short hand for cost function
costFunc = @(p) sparseAutoencoderCost(p, input_layer_size, hidden_layer_size, ...
                                             lambda, sparsityParam, beta, data);                            

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);
                               
% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');

end
