%% Kiambe Kevin ASSA
% Instite of Science and Technology
% Computer Engineering Department
% Selçuk University
% Konya, Turkey
% Copyright 2018 Selçuk Üniversitesi
% Part 2
% Feature extraction using the trained CNN for the recognition task
%% Step 1: Check system requirements
% 1-Neural Network Toolbox(TM),
% 2-Statistics and Machine Learning Toolbox(TM),
% 3-Neural Network Toolbox(TM) Model _for AlexNet Network.
% 4-Computer Vision System Toolbox

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

rootFolder = 'CnnsvmImageSet';
% Create infection data categories for the 4 classes of breast cancer
categories = {'Normal','Benign','Invasive','InSitu'};
% Create an ImageDatastore to help you manage the image data.
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%% Step 3: Load the trained CNN Network
% Load pre-trained AlexNet
load('DeepHist.mat', 'trainedCnnNet');
net = trainedCnnNet;
% View the CNN architecture
net.Layers
% Inspect the first layer
net.Layers(1)
% Inspect the last layer
net.Layers(end)
% Number of class names for ICIAR2018 BACH classification task
numel(net.Layers(end).ClassNames)

%% Step 4: Pre-process Images For CNN by resizing the images
imgSize=227; % image resize dimension for input layer
imds.ReadFcn = @(loc)imresize(imread(loc),[imgSize,imgSize]);

%% Step 5: Prepare Training and Test Image Sets
% Split the sets into training and validation data. For example a
% 0.3 split will create 30% of the images for training and the rest for validation
% Randomize the split to avoid biasing the results.
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

%% Step 6: Extract Training Features Using CNN
% Extracting training features from the deep CNN neural network layer.
featureLayer = 'fcn1';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Step 7: Train A Multiclass SVM Classifier Using CNN Features
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;
% Train classical machine learning classifier. 
% Uncomment the classifier
t=templateLinear();
% t=templateLinear('Learner','svm');
% t=templateLinear('Learner','logistic');
% t=templateKNN('Distance','euclidean');

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', t, 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Step 8: Evaluate Classifier
% Extract test features using the CNN
testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);
% Get the known labels
testLabels = testSet.Labels;
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
% Display the mean accuracy
mean(diag(confMat))
%% Step 9:  Try the Newly Trained Classifier on Test Images
newImage = fullfile(rootFolder, 'Normal', 'CCFH_00015.jpg');
% Pre-process the images as required for the CNN
img = readAndPreprocessImage(newImage);
% Extract image features using the CNN
imageFeatures = activations(net, img, featureLayer);
% Make a prediction using the classifier
label = predict(classifier, imageFeatures);
function Iout = readAndPreprocessImage(filename)
    I = imread(filename);
    % Resize the image as required for the CNN.
    Iout = imresize(I, [227 227]);
end
%% References

% As indicated in the article