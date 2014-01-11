function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% size(Theta1_grad)
% size(Theta2_grad)
X = [ones(m, 1), X];
a1 = X;
z2 = a1*Theta1';
% size(z2)
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);
tmp = (repmat(y, [1, num_labels]) == repmat(1:num_labels, [m, 1]));
J = sum(sum(-tmp.*log(a3) - (1-tmp).*log(1-a3), 2))/m + lambda*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:, 2:end).^2, 2)))/(2*m);

delta_3 = zeros(num_labels, 1);
for i = 1:m
    delta_3(:) = a3(i, :) - tmp(i, :);
    delta_2 = (Theta2(:, 2:end)'*delta_3).*((sigmoidGradient(z2(i, :)))');
    Theta2_grad = Theta2_grad + delta_3*a2(i, :);
    Theta1_grad = Theta1_grad + delta_2*a1(i, :);
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda*Theta1(:, 2:end)/m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda*Theta2(:, 2:end)/m;
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
