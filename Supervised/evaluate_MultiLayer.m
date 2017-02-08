%% This script will help you test out your single layer neural network code
rng(123);
%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};
%% Modify the X Matrices so that a bias is added

Xt{1}=Xt{1}-mean(Xt{1}(:));
Xt{1}=Xt{1}/std(Xt{1}(:));

% The Training Data
Xtraining = [ones(1,size(Xt{1},2));Xt{1}]; % Remove this line

Xt{2}=Xt{2}-mean(Xt{2}(:));
Xt{2}=Xt{2}/std(Xt{2}(:));
% The Test Data
Xtest = [ones(1,size(Xt{2},2));Xt{2}]; % Remove this line


%% Train your single layer network
% Note: You nned to modify trainSingleLayer() in order to train the network

numHidden = 130 % Change this, Number of hidde neurons 
numIterations = 10000 % Change this, Numner of iterations (Epochs)
learningRate = 0.021 % Change this, Your learningrate
W = normrnd(0, 1, [numHidden size(X, 1)]); % Change this, Initiate your weight matrix W
V = normrnd(0, 1, [size(D, 1) numHidden]); % Change this, Initiate your weight matrix V
totalIterations = numIterations;
%totalIterations = totalIterations + numIterations;
%
tic
[W, V, trainingError, testError ] = trainMultiLayer(Xtraining, Dt{1}, Xtest, Dt{2}, W, V, numIterations, learningRate);
trainingTime = toc;
%% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);
plot(trainingError,'k','linewidth',1.5)
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Multi-Layer')
legend('Training Error','Test Error','Min Test Error')

%% Calculate The Confusion Matrix and the Accuracy of the Evaluation Data
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

[ Y, ~, LMultiLayerTraining ] = runMultiLayer(Xtraining, W, V);
tic
[ Y, ~, LMultiLayerTest ] = runMultiLayer(Xtest, W,V);
classificationTime = toc/length(Xtest);
% The confucionMatrix
cM = calcConfusionMatrix( LMultiLayerTest, Lt{2});

% The accuracy
acc = calcAccuracy(cM);

display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Time spent calssifying 1 feature vector: ' num2str(classificationTime) ' sec'])
display(['Accuracy: ' num2str(acc)])

%% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultMultiLayer(W,V,Xtraining,Lt{1},LMultiLayerTraining,Xtest,Lt{2},LMultiLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LMultiLayerTest )
end
