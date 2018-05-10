clear%% Kiambe Kevin ASSA
% Instite of Science and Technology
% Computer Engineering Department
% Selçuk University
% Konya, Turkey  
% Copyright 2018 Selçuk Üniversitesi
% Part 1
%This is the first section for training the CNN model
%% Step 1: Check system requirements
% 1-Neural Network Toolbox(TM), 
% 2-Statistics and Machine Learning Toolbox(TM),
% 3-Neural Network Toolbox(TM) Model _for AlexNet Network.
% 4-Computer Vision System Toolbox
%
% Recommend using a CUDA-capable NVIDIA(TM) GPU with compute capability 3.0 or higher
% Get GPU device information
deviceInfo = gpuDevice;
% Check the GPU compute capability
computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability >= 3.0, ...
    'GPU device compute capability must be 3.0 or higher.')
%% Step 2: Load data and set up training data
% Load the 7200 image augemented image dataset 
% 4 categories of the breast cancer type(normal, benign, in situ carcinoma
% and invasive carcinoma)from the ICIAR Image dataset 2017 (https://iciar2018-challenge.grand-challenge.org/dataset/)
rootFolder = 'RunFold4/Train';
categories = {'Normal','Benign','Invasive','InSitu'};
imgSize=227; % image resize dimension for input layer
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(loc)imresize(imread(loc),[imgSize,imgSize]);
[trainingImages,validationImages] = splitEachLabel(imds,0.9,'randomized');

%% Step 3: Set up Hyperparameters
miniBatchSize = 20; 
lr=1e-3; % Learning rate
maxEpoch=1000; %Maximum number of epoch

%% Step 3: Transfer learning using Matlab AlexNet add-on and visualize layers
extractNetModel = alexnet;
layers = extractNetModel.Layers;
disp(layers);

%% Step 5: Create the Convolution Neural Network(CNN)
layers = layers(1:end-3); %Remove last 1000 class classification layer
%Add a 2 fully connected layers with rectifier linear units and maximum
%pooling layers to make the network deeper
layers(end+1) = fullyConnectedLayer(4096,'BiasLearnRateFactor',2, 'WeightLearnRateFactor',1, 'WeightL2Factor',1, 'BiasL2Factor',0,'Name','fcn1' );
layers(end+1) = reluLayer;
maxPooling2dLayer(3,'Stride',2);
layers(end+1) = fullyConnectedLayer(4,'BiasLearnRateFactor',2, 'WeightLearnRateFactor',1, 'WeightL2Factor',1, 'BiasL2Factor',0,'Name','fcn2' );
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer();

% Setup training options for our CNN model
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize); %Number of iterations per epoch
tOptions = trainingOptions('sgdm', ...
    'InitialLearnRate', lr, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', maxEpoch, ...
    'MiniBatchSize',miniBatchSize,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch,...
    'Verbose', false);

%% Step 6: Start training the network using the image set
% Preferably training with a GPU is strongly encouraged.
[trainedCnnNet, info] = trainNetwork(trainingImages, layers, tOptions);
trainedNetlayers = trainedCnnNet.Layers;
disp(trainedNetlayers);
%% Step 7: Validation of the trained network
%Classify new image set using the validation image data set
predictedLabels = classify(trainedCnnNet,validationImages);
%Accuracy
valLabels = validationImages.Labels;
trainingAccuracy = mean(predictedLabels == valLabels);
displayTrainingAccuracy = sprintf('%f ##### Training accuracy.',(trainingAccuracy*100));
disp(displayTrainingAccuracy);

%% Step 8: Testing the trained network with new image data that was not included in the training data set 
% Load test data
rootFolder = 'RunFold4/Test20';
testData = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testData.ReadFcn = @(loc)imresize(imread(loc),[imgSize,imgSize]);

% Test classifer for the trained network
[predictedInfection,err_test] = classify(trainedCnnNet, testData);

% Display random set of image with their predicted classes
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(testData,idx(i));
    label = predictedInfection(idx(i));
    imshow(I)
    title(char(label))
    
    ii = randi(4);
    im = imread(testData.Files{ii});
    imshow(im);
    if predictedInfection(ii) == testData.Labels(ii)
        colorText = 'b';
    else
        colorText = 'r';
    end
    title(char(predictedInfection(ii)),'Color',colorText);
end


% Determine overall testing accuracy
actualInfection = testData.Labels;
testAccuracy = mean(predictedInfection == actualInfection);
displayTestAccuracy = sprintf('%f #####  Testing accuracy.',(testAccuracy*100));
disp(displayTestAccuracy);

%% Step 9: Save the model for secondary storagy
save('DeepHist.mat', 'trainedCnnNet');
