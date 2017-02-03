function [Wout, trainingError, testError ] = trainSingleLayer(Xt,Dt,Xtest,Dtest, W0,numIterations, learningRate )
%TRAINSINGLELAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               W0 - Weights of the neurons (matrix)
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
Nt = size(Xt,2);
Ntest = size(Xtest,2);
Wout = W0;

% Calculate initial error
Yt = runSingleLayer(Xt, W0);
Ytest = runSingleLayer(Xtest, W0);
trainingError(1) = sum(sum((Yt - Dt).^2))/Nt;
testError(1) = sum(sum((Ytest - Dtest).^2))/Ntest;

Dt
Dtest

for n = 1:numIterations
    Yt = runSingleLayer(Xt, Wout);
    Ytest = runSingleLayer(Xtest, Wout);
    %Y = Wout*Xt;
    
    %size(Dt)
    %size(Yt)
    %size(Xt)
    
    grad_w = -(Dt - Yt)*Yt'*(1 - Yt)*Ytest';
    
    %size(grad_w)

    Wout = Wout - learningRate*grad_w;
    %size(Wout)
    
    trainingError(1+n) = sum(sum((Wout*Xt(2:end,:) - Dt).^2))/Nt;
    testError(1+n) = sum(sum((Wout*Xtest(2:end,:) - Dtest).^2))/Ntest;
end
end

