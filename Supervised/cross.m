%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data
rng(12345)
dataSetNr = 4; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );
L = L';

kRange = 1:15;
kFold = 5;
bestK = 1;
bestAccuracy = 0;

indices = crossvalind('Kfold', size(X,2), kFold);

for k = kRange
    currentAccuracy = 0;
    for fold = 1:kFold
        test = X(:,find(indices==fold));
        test_labels = L(:,find(indices==fold));
        
        train = X(:,find(indices~=fold));
        train_labels = L(:,find(indices~=fold));
        
        LkNN = kNN(test, k, train, train_labels);
        cM = calcConfusionMatrix(LkNN, test_labels);
        
        
        currentAccuracy = (currentAccuracy * (fold - 1) + calcAccuracy(cM)) / (fold);
        
    end
    
    if bestAccuracy < currentAccuracy
        bestK = k;
        bestAccuracy = currentAccuracy;
    end
end

%plotkNNResultDots(Xt{2},LkNN,k,Lt{2},Xt{1},Lt{1});
