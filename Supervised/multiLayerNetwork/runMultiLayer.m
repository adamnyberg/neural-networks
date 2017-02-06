function [ Y, U, L ] = runMultiLayer( X, W, V )
%RUNMULTILAYER Calculates output and labels of the net
%   Inputs:
%               X  - Features to be classified (matrix)
%               W  - Weights of the hidden neurons (matrix)
%               V  - Weights of the output neurons (matrix)
%
%   Output:
%               Y = Output for each feature, (matrix)
%               L = The resulting label of each feature, (vector) 

S = 0; %Calculate the sumation of the weights and the input signals (hidden neuron)
U = 0; %Calculate the activation function as a hyperbolic tangent

U = tanh((W*X(2:end,:))+X(1,:)); %Calculate the sumation of the output neuron
Y = tanh((V*U)+X(1,:));
   

% Calculate classified labels
[~, L] = max(Y,[],1);
L = L(:);

end

