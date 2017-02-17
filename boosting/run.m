% Load face and non-face data and plot a few examples
load faces, load nonfaces
faces = double(faces);
nonfaces = double(nonfaces);


nbrHaarFeatures = 160;
nbrTrainExamples = 400;
nbrTestExamples = 3000;
nbrClassifiers = 50;
epsLim = 0.0001;
defaultAlpha = 25;

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

for t = 1:T
    eps = zeros(M, N);
    pol = ones(M, N);
    
    for m = 1:M % features
        for n = 1:N % thresholds
            currentThreshold = xTrain(m, n);
            
            xThreshold = +(xTrain(m,:) < currentThreshold);
            xThreshold(xThreshold == 0) = -1;
            I = +(xThreshold ~= yTrain);
            
            eps(m, n) = sum(d.*I);
            
            if eps(m, n) > 0.5
                eps(m, n) = 1 - eps(m, n);
                pol(m, n) = -1;
            end
        end
    end
    
    [feature threshold] = find(eps == min(eps(:))); %Finds all the best
    
    feature = feature(1);
    threshold = threshold(1);
    epsilon = eps(feature, threshold);
    polarity = pol(feature, threshold);
    
    if epsilon > epsLim
        alpha = 0.5*log((1-epsilon)/epsilon);
    else
        alpha = defaultAlpha;
    end
    
    classifiers(:,t) = [epsilon; feature; threshold; polarity; alpha]; % save best classifier
    
    xThreshold = +(xTrain(feature,:) < threshold);
    xThreshold(xThreshold == 0) = -1;
    I = +(xThreshold ~= yTrain);
    
    I(I == 0) = -1;
    if pol(m, n) < 0
        misClass = ~I;
    else
        misClass = I;
    end
    
    d = d.*exp(-alpha * misClass);
    
    d = max(0.1/N, d); % Set max limit
    d = min(10/N, d); % Set min limit
    
    d = d/sum(d); % Normalize
end

correct = 0;
h1 = 0;

for i = 1:Ntest
    
    h1 = strong(classifiers, xTest(:,i), T);
    
    if h1 == yTest(1,i)
        correct = correct +1;   
    end
    
end

acc = correct/Ntest;


trainingTime = toc;
display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Accuracy: ' num2str(acc)])
            
    
        
        
function h = strong(H, X, T)

    h = 0;

    for i = 1:T
        % Weak classifiers features
        epsilon = H(1,i);
        feature = H(2,i);
        threshold = H(3,i);
        polarity = H(4,i);
        alpha = H(5,i);
        x = X(feature);

        if x*polarity < threshold*polarity
           h = h + alpha;
        else
           h = h - alpha;
        end
    end

    h = sign(h);
end







