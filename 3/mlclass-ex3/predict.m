function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1), X];
a = [ones(m, 1), sigmoid(X*Theta1')];
a = sigmoid(a*Theta2');
[~, p] = max(a, [], 2);
end
