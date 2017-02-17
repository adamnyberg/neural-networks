% Load face and non-face data and plot a few examples
load faces, load nonfaces
faces = double(faces);
nonfaces = double(nonfaces);
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

nbrHaarFeatures = 25;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
    axis image,axis off
end



% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 100;

trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));

xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];


N = nbrTrainExamples*2;
M = nbrHaarFeatures;
T = nbrHaarFeatures;

d = 1/N*ones(1, M);
classifiers = zeros(5, T);

for t = 1:T
    eps = zeros(M, N);
    pol = ones(M, N);
    misClass = zeros(M, N)
    
    
    for m = 1:M % features
        for n = 1:N % thresholds
            currentThreshold = xTrain(m, n);
            
            xThreshold = +(xTrain(f,:) < currentThreshold);
            xThreshold(xThreshold == 0) = -1;
            I = +(xThreshold ~= yTrain);
            
            eps(m, n) = sum(d.*I);
            
            if eps(m, n) < 0.5
                eps(m, n) = 1 - eps(m, n);
                pol(m, n) = -1;
            end
            
            I(I == 0) = -1;
            if p < 0
                misClass(M, N) = ~I;
            else
                misClass(M, N) = I;
            end
        end
    end
    
    [feature threshold] = find(eps == min(eps(:))); %Finds all the best
    
    feature = feature(1); 
    threshold = threshold(1);
    epsilon = eps(feature, threshold);
    polarity = pol(feature, threshold);
    misclassification = misClass(feature, threshold);
    alpha = 0.5*log((1-epsilon)/epsilon);
    
    classifiers(:,t) = [feature; threshold; epsilon; polarity; alpha]; % save best classifier
    
    d = d.*exp(-alpha * misClassification);
    
    d = max(0.1/(nbrTrainExamples*2), d); % Set max limit
    d = min(10/(nbrTrainExamples*2), d); % Set min limit
    d = d/sum(d); % Normalize
end


            
            
    
        
        







