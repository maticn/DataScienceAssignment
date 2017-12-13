%% Load Images
rootFolder = fullfile(pwd, 'images');
categories = {'manet', 'notManet'};
% ImageDatastore operates on image file locations and help you manage the data.
images = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
% Summarize the number of images per category.
numOfImagesPerCategory = countEachLabel(images)  


%% Balance the number of images in the training set
minNumOfImagesPerCategory = min(numOfImagesPerCategory{:,2}); % determine the smallest amount of images in a category
% Use splitEachLabel method to trim the set.
images = splitEachLabel(images, minNumOfImagesPerCategory, 'randomize');
% Notice that each set now has exactly the same number of images.
countEachLabel(images)


%% Load Pre-trained AlexNet Network
net = alexnet()

%% Pre-process Images For CNN
% net can only process RGB images that are 227-by-227.
% imds.ReadFcn pre-process images on-the-fly.
% The imds.ReadFcn is called every time an image is read from the ImageDatastore.
images.ReadFcn = @(filename)readAndPreprocessImage(filename);


%% Prepare Training and Test Image Sets
% Pick 75% of images from each set for the training data and the remainder, 25%, for the validation data.
% Randomize the split to avoid biasing the results.
[train, test] = splitEachLabel(images, 0.75, 'randomize');


%% Extract Training Features Using CNN
% Extract features from one of the deeper layers using the activations method.
% Which of the deep layers to choose? Typically starting with the layer right before the classification layer.
% In net, this layer is named 'fc7'. Let's extract training features using that layer.
featureLayer = 'fc7';
trainingFeatures = activations(net, train, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');


%% Train A Multiclass SVM Classifier Using CNN Features
% Use the CNN image features to train a multiclass SVM classifier.

% Get training labels from the trainingSet
trainLabels = train.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


%% Evaluate Classifier
% Extract image features from test set using the CNN
testFeatures = activations(net, test, featureLayer, 'MiniBatchSize', 32);

% Pass CNN image features to trained classifier
[predictedLabels] = predict(classifier, testFeatures);

% Get the known labels
testLabels = test.Labels;

% Tabulate the results using a confusion matrix.
confusionMatrix = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confusionMatrix = bsxfun(@rdivide, confusionMatrix, sum(confusionMatrix, 2))

% Display the mean accuracy
mean(diag(confusionMatrix))


%% Try the Newly Trained Classifier on Test Images

% Load test images
testImagesFolder = fullfile(rootFolder, 'testImages');
d = dir([testImagesFolder, '\*.bmp']);
fileNames = char(d.name);

% Classify all files.
for idx = 1:length(d)
    newImage = fullfile(testImagesFolder, fileNames(idx, :))

    % Pre-process the images as required for the CNN
    img = readAndPreprocessImage(newImage);

    % Extract image features using the CNN
    imageFeatures = activations(net, img, featureLayer);

    % Make a prediction using the classifier
    label = predict(classifier, imageFeatures)
end


%% Helper Functions
function preparedImage = readAndPreprocessImage(file)
    img = imread(file);

    % Some images may be grayscale. Replicate the image 3 times to create an RGB image.
    if ismatrix(img)
        img = cat(3, img, img, img);
    end

    % Resize the image as required for the CNN.
    preparedImage = imresize(img, [227 227]);
end
