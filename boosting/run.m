
% Load face and non-face data and plot a few examples
load faces, load nonfaces
faces = double(faces);
nonfaces = double(nonfaces);
%rng(12345)

nbrHaarFeatures = 100;
nbrTrainExamples = 100;
nbrTestExamples = 3000;
nbrClassifiers = 200;
epsLim = 0.0001;
defaultAlpha = 5;

tic

figure(1)
colormap gray
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k)), axis image, axis off
end
figure(2)
colormap gray
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k)), axis image, axis off
end
% Generate Haar feature masks

haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
    axis image,axis off
end



% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1

trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

testImages = cat(3,faces(:,:,nbrTrainExamples:nbrTrainExamples+nbrTestExamples),nonfaces(:,:,nbrTrainExamples:nbrTrainExamples+nbrTestExamples));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

N = nbrTrainExamples*2;
Ntest = nbrTestExamples*2;
M = nbrHaarFeatures;
T = nbrClassifiers;

d = 1/N*ones(1, N);

% For every classifier..
for t = 1:T
    eps = zeros(M, N);
    pol = ones(M, N);
    
    for m = 1:M % ... test every feature ...
        for n = 1:N % ... with every possible threshold
                       
            % Extract current threshold
            currentThreshold = xTrain(m, n);
            
            % Check how well the current theshold performs
            xThreshold = +(xTrain(m,:) < currentThreshold);
            xThreshold(xThreshold == 0) = -1;
            I = +(xThreshold ~= yTrain);
            
            % Calculate the error
            eps(m, n) = sum(d.*I);
            
            % If the error is too large, flip the polarity
            if eps(m, n) > 0.5
                eps(m, n) = 1 - eps(m, n);
                pol(m, n) = -1;
            end
        end
    end
    
    %Finds the best feature and threshold
    [feature threshold] = find(eps == min(eps(:))); 
    feature = feature(1);
    threshold = threshold(1);
    
    tao = xTrain(feature, threshold);
    epsilon = eps(feature, threshold);
    polarity = pol(feature, threshold);
    
    if eps > epsLim
        alpha = 0.5*log((1-epsilon)/epsilon);
    else
        alpha = defaultAlpha;
    end
    
    classifiers(:,t) = [epsilon; feature; tao; polarity; alpha]; % save best classifier    
    
    error = ((xTrain(feature,:) < tao) & (yTrain == 1)) | ((xTrain(feature,:) >= tao) & (yTrain == -1));
    if(polarity == -1)
        error = ~error;
    end
    error = 2*error; 
    misclass = error - 1;
    
    d = d.*exp(-alpha .* misclass);
    d = max(0.1/N, d); % Set max limit

    d = min(10/N, d); % Set min limit
    d = d/sum(d); % Normalize
end

acc = accuracy(classifiers, xTest, yTest);


trainingTime = toc;
display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Accuracy: ' num2str(acc)])


%%
accs = [];
for t = 1:T 
    accs(t) = accuracy(classifiers(:,1:t), xTest, yTest);
end

bestAcc = max(accs);
bestNumberOfClassifiers = find(max(accs)==accs);

figure(4)
plot(accs);
xlabel('# of week classifiers', 'FontSize', 16);
ylabel('Accuracy', 'FontSize', 16);

    
        



