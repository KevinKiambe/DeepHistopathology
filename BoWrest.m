%% Train SVM Classifier
% opts = templateSVM('KernelFunction', 'rbf');
opts = templateSVM('KernelFunction', 'linear');
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
% bagTest = bagOfFeatures(imgSetTest, 'StrongestFeatures', ratioOfStrongFeatures, 'VocabularySize', numberOfClusters);
confMatrix_test = evaluate(classifier, imgSetTest);
test_accuracy = mean(diag(confMatrix_test));
display(['The test accuracy is ' test_accuracy '.']);


%% Visualize feature vector of the first image in the test set
