function [Wout,Vout, trainingError, testError ] = trainMultiLayer(Xt,Dt,Xtest,Dtest, W0, V0,numIterations, learningRate )
%TRAINMULTILAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               V0 - Weights of the output neurons (matrix)
%               W0 - Weights of the output neurons (matrix)
%               numIterations - Number of learning setps (scalar)
%               learningRate - The learningrate (scalar)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
numTraining = size(Xt,2);
N = numTraining;
numTest = size(Xtest,2);
numClasses = size(Dt,1) - 1;
Wout = W0;
Vout = V0;

% Calculate initial error
Yt = runMultiLayer(Xt, W0, V0);
Ytest = runMultiLayer(Xtest, W0, V0);
trainingError(1) = sum(sum((Yt - Dt).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);

for n = 1:numIterations
    [Yt, Ut, ~] = runMultiLayer(Xt, Wout, Vout);
    %size(Ut)

    grad_v = (2/N) * (Vout*Ut - Dt) * Ut'; %Calculate the gradient for the output layer
    
    grad_w = (2/N) * (Vout'*(Yt - Dt)) .* (ones(size(Ut))-Ut.^2) * Xt(2:end,:)'; %..and for the hidden layer.
   
    %size(grad_v)
    %size(grad_w)
    
    Vout = Vout - learningRate * grad_v; %Take the learning step.
    Wout = Wout - learningRate * grad_w; %Take the learning step.
    
    %size(Vout)
    %size(Wout)

    Yt = runMultiLayer(Xt, Wout, Vout);
    Ytest = runMultiLayer(Xtest, Wout, Vout);

    trainingError(1+n) = sum(sum((Yt - Dt).^2))/(numTraining*numClasses);
    testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
end

end

