%% Kiambe Kevin ASSA
% Instite of Science and Technology
% Computer Engineering Department
% Selçuk University
% Konya, Turkey
% Copyright 2018 Selçuk Üniversitesi
% Part 3
% Testing 
load('cNNSVMModel.mat', 'net');
% Get list of all image files from the directory
fileFolder = fullfile('Preprocessed/CroppedRight');% Specify the cropped bach to be used. 
% Uncomment to use other crops
% fileFolder = fullfile('Preprocessed/CroppedCenter');
% fileFolder = fullfile('Preprocessed/CroppedLeft');
imageFiles = dir(fullfile(fileFolder,'*.jpg'));
for ii=1:length(imageFiles)
   currentfilename = imageFiles(ii).name;
   newTestImage = fullfile(fileFolder, currentfilename);
   % Pre-process the images as required for the CNN
   img = readAndPreprocessImage(newTestImage);
   % Extract image features using the CNN
   newImageFeatures = activations(net, img, featureLayer);
   % Make a prediction using the classifier
   label = predict(classifier, newImageFeatures);
   %Display the predicted label
   x = sprintf('Predicted Label for image %s is  %s',currentfilename,label);
   disp(x);
end

function Iout = readAndPreprocessImage(filename)
    I = imread(filename);
    % Resize the image as required for the CNN.
    Iout = imresize(I, [227 227]);
end