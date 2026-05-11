% MATLAB CNN Model Training and Evaluation
clear; close all;

% Step 1: Load and Prepare Data
[X, y] = load_and_prepare_data('LSTMdata.mat');
% 数据集划分为训练集和验证集
numTrain = floor(0.8 * size(X, 1));
XTrain = X(1:numTrain, :);
yTrainSeq = categorical(y(1:numTrain));
XVal = X(numTrain+1:end, :);
yValSeq = categorical(y(numTrain+1:end));

%%
% Step 2: Define CNN Architecture
layers = [
    sequenceInputLayer(size(X, 2), 'Name', 'input', 'MinLength', size(X, 2)) % 设置 MinLength 为输入特征数
    
    convolution1dLayer(3, 128, 'Padding', 'same', 'Name', 'conv1') % 一维卷积
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1') % 一维最大池化

    convolution1dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2') % 第二个卷积层
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2') % 第二个池化层

    dropoutLayer(0.5, 'Name', 'dropout')

    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.5, 'Name', 'dropout2')

    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Step 3: Define Training Options
options = trainingOptions('rmsprop', ...
    'MaxEpochs', 200, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.5, ...
    'Verbose', 1, ...
    'MiniBatchSize', 64, ...
    'Plots', 'training-progress', ...
    'ValidationData', {XVal, yValSeq}, ...
    'ValidationFrequency', floor(numTrain / 64));


% Step 4: Train the Network
net = trainNetwork(XTrain, yTrainSeq, layers, options);

% Step 5: Predict and Evaluate
YPredProb = predict(net, XVal, 'MiniBatchSize', 64);

% Convert probabilities to categories
[~, YPred] = max(YPredProb, [], 2); 
YPred = categorical(YPred - 1);

% Calculate accuracy
accuracy = sum(YPred == yValSeq) / numel(yValSeq);
fprintf('Validation Accuracy: %.4f\n', accuracy);

% Plot confusion matrix
figure;
confusionchart(yValSeq, YPred);
title('Validation Confusion Matrix');

% Step 6: Evaluate on Training Data
YPredTrainProb = predict(net, XTrain, 'MiniBatchSize', 64);
[~, YPredTrain] = max(YPredTrainProb, [], 2);
YPredTrain = categorical(YPredTrain - 1);

figure;
confusionchart(yTrainSeq, YPredTrain);
title('Train Confusion Matrix');
