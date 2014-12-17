function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------     
	
	m = size(patches, 2);
	activations = weightMatrix * patches;
    inside = sqrt(activations .^ 2 + epsilon);
    cost = 1/m * sum(inside(:));
    
    %calculate grad
    grad = 1/m * (activations ./ inside)*patches';
	grad = grad(:);
	
end