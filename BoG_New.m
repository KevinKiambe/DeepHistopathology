%% BoW
classes = {'Benign', 'InSitu' , 'Invasive', 'Normal'}; % Should be sorted alphapetically to match test data automatic labeling by folder
trainingSplit=0.9;
numberOfClusters = 500; % The number of clusters representing the number of features in the bag of features. Default can be 500.
ratioOfStrongFeatures = 0.8; % Default can be 0.8

%% Load Image Datastore and make them of equal size (size of class with lowest number of images)
imgSets = [];
for i = 1:length(classes)
    imgSets = [ imgSets, imageSet(fullfile('Fold5/Train', classes{i})) ];
end

% Balance the data count between of all classes
minClassCount = min([imgSets.Count]);
imgSets = partition(imgSets, minClassCount, 'sequential'); 

%% Prepare Training and Validation Image Sets
[trainingSets, validationSets] = partition(imgSets, trainingSplit, 'sequential'); 

%% Visulizing Training and Validation Data (Press any key to procceed to next figure)
% mkdir('Graphs//'); %Directory for Storing Results

%% Forming Bag of Features
% Extracts SURF features from all training images &
% reducing the number of features through quantization of feature space using K-means clustering
bagTrain = bagOfFeatures(trainingSets, 'StrongestFeatures', ratioOfStrongFeatures, 'VocabularySize', numberOfClusters);
% 
%% Visualize feature vector of the first image in the train set

figure;
imgTrain = read(imgSets(1), 1); % First image of first class as an example
featureVector = encode(bagTrain, imgTrain);

% Plot the histogram of visual word occurrences
figure;
bar(featureVector);
title('Gorsel Kelime Olusumlari');
xlabel('Gorsel Kelime Indeksi');
ylabel('Olusma Frekansi');
saveas(gcf,['Graphs//FeatureVectorTrain5.png']);

%% Train SVM Classifier
opts = templateSVM('KernelFunction', 'rbf');
% opts = templateSVM('KernelFunction', 'linear');
% opts = templateSVM('KernelFunction', 'knn');
% classifier = trainImageCategoryClassifier(trainingSets, bag, 'LearnerOptions', opts);
classifier = trainImageCategoryClassifier(trainingSets, bagTrain, 'LearnerOptions', opts);   
%% Evaluate the classifier on training then validation data
confMatrix_train = evaluate(classifier, trainingSets);
confMatrix_val = evaluate(classifier, validationSets);
train_accuracy = mean(diag(confMatrix_train));
validation_accuracy = mean(diag(confMatrix_val));
display(['The training accuracy is ' train_accuracy '.']);
display(['The validation accuracy is ' validation_accuracy '.']);
%% Deployment (test the system on newly unseen images)

imgSetTest = [];
for i = 1:length(classes)
    imgSetTest = [ imgSetTest, imageSet(fullfile('Fold5/Test20', classes{i})) ];
end
bagTest = bagOfFeatures(imgSetTest, 'StrongestFeatures', ratioOfStrongFeatures, 'VocabularySize', numberOfClusters);
confMatrix_test = evaluate(classifier, imgSetTest);
test_accuracy = mean(diag(confMatrix_test));
display(['The test accuracy is ' test_accuracy '.']);


%% Visualize feature vector of the first image in the test set

figure;
imgTest = read(imgSetTest(1), 1); % First image of first class as an example
featureVector = encode(bagTest, imgTest);

% Plot the histogram of visual word occurrences
figure;
bar(featureVector);
title('Gorsel Kelime Olusumlari');
xlabel('Gorsel Kelime Indeksi');
ylabel('Olusma Frekansi');
saveas(gcf,['Graphs//FeatureVectorTest5.png']);
