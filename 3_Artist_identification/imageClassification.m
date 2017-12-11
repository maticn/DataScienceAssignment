%% Load Images
rootFolder = fullfile(pwd, 'images');
categories = {'manet', 'degas', 'renoir', 'monet'};
% ImageDatastore operates on image file locations and help you manage the data.
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
% Summarize the number of images per category.
tbl = countEachLabel(imds)  


%% Balance the number of images in the training set
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Notice that each set now has exactly the same number of images.
countEachLabel(imds)


%% Load Pre-trained AlexNet Network
net = alexnet()

%% Pre-process Images For CNN
% net can only process RGB images that are 227-by-227.
% imds.ReadFcn pre-process images on-the-fly.
% The imds.ReadFcn is called every time an image is read from the ImageDatastore.
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);


%% Prepare Training and Test Image Sets
% Pick 30% of images from each set for the training data and the remainder, 70%, for the validation data.
% Randomize the split to avoid biasing the results.
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');


%% Extract Training Features Using CNN
% Extract features from one of the deeper layers using the activations method.
% Which of the deep layers to choose? Typically starting with the layer right before the classification layer.
% In net, this layer is named 'fc7'. Let's extract training features using that layer.
featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');


%% Train A Multiclass SVM Classifier Using CNN Features
% Use the CNN image features to train a multiclass SVM classifier.

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


%% Evaluate Classifier
% Extract image features from testSet using the CNN
testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize', 32);

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
function Iout = readAndPreprocessImage(filename)
    I = imread(filename);

    % Some images may be grayscale. Replicate the image 3 times to create an RGB image.
    if ismatrix(I)
        I = cat(3,I,I,I);
    end

    % Resize the image as required for the CNN.
    Iout = imresize(I, [227 227]);
end
