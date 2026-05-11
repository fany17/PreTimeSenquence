% MATLAB CNN Model Training and Evaluation
clear; close all;

% Step 1: Load and Prepare Data
[X, y] = load_and_prepare_data('LSTMdata.mat');

% 数据集信息
% 时间步数: 570182
% X: 570182x13 double
% y: 570182x1 double (二分类指标，包含0和1)

% Step 2: 设置窗口大小，划分数据为序列段
windowSize = 20000; % 定义窗口大小
seglength = 50;
numSegments = floor((size(X, 1) - seglength) / windowSize); % 计算序列段数量

XSeq = {};
ySeq = [];

% 使用滑动窗口创建序列段，每个cell包含一个序列段
for i = 1:numSegments
    startIdx = (i - 1) * windowSize + 1;
    endIdx = startIdx + seglength - 1;
    XSegment = X(startIdx:endIdx, :);
%     XSegment1 = X(startIdx:endIdx, :);
%     XSegment2 = X(startIdx+ seglength - 1:endIdx+ seglength - 1, :);
%     XSegment = [XSegment1 XSegment2];
    XSegment = mapminmax(XSegment',0,1)';    
    XSeq{end+1} = reshape(XSegment, [seglength, size(X, 2), 1]); % 转换为2D张量
    ySeq(end+1) = y(endIdx); % 每段的标签为最后一个时间步的标签
end
ySeq = categorical(ySeq);

%%
% Step 3: 随机划分训练集和验证集
trainRatio = 0.8;
numTrain = floor(trainRatio * numSegments);

% 随机打乱序列段索引
randomIdx = randperm(numSegments);
trainIdx = randomIdx(1:numTrain);
valIdx = randomIdx(numTrain+1:end);

% 分配训练集和验证集
XTrainSeq = XSeq(trainIdx);
yTrainSeq = ySeq(trainIdx);
XValSeq = XSeq(valIdx);
yValSeq = ySeq(valIdx);

% 转换为4D数组
XTrain = cat(4, XTrainSeq{:});
XVal = cat(4, XValSeq{:});

% Step 4: 定义CNN网络架构% 构建一个层次图
% 定义 CNN 网络架构
layers = [
    imageInputLayer([seglength, size(X, 2), 1], 'Name', 'input')
    
    convolution2dLayer([5,5], 64, 'Padding', 'same', 'Name', 'large_conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2, 2], 'Stride', [2, 2], 'Name', 'pool1')
    dropoutLayer(0.5, 'Name', 'dropout1')

    convolution2dLayer([3, 3], 128, 'Padding', 'same', 'Name', 'small_conv1')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2, 2], 'Stride', [2, 2], 'Name', 'pool2')
    dropoutLayer(0.5, 'Name', 'dropout2')
    
    convolution2dLayer([1, 1], 256, 'Padding', 'same', 'Name', 'small_conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2, 2], 'Stride', [2, 2], 'Name', 'pool3')
    dropoutLayer(0.5, 'Name', 'dropout3')

%     % 添加 LSTM 层
%     flattenLayer('Name', 'flatten')
%     lstmLayer(256, 'OutputMode', 'last', 'Name', 'lstm')
%     dropoutLayer(0.5, 'Name', 'dropout_lstm')

    % 减少全连接层的节点数
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout4')

    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];


% Step 5: 定义训练选项
options = trainingOptions('rmsprop', ...
    'MaxEpochs', 2000, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.5, ...
    'Verbose', 0, ...
    'MiniBatchSize', 64, ... % 设置 batch size，例如 64
    'Plots', 'training-progress', ...
    'ValidationData', {XVal, yValSeq}, ...
    'ValidationFrequency', floor(numTrain/5));


% Step 6: 训练网络
net = trainNetwork(XTrain, yTrainSeq, layers, options);
% % Step 6: 训练网络，注意使用 lgraph 而非 layers
% net = trainNetwork(XTrain, yTrainSeq, lgraph, options);


% Step 7: 进行预测
YPredProb = predict(net, XVal, 'MiniBatchSize', 1); % 预测概率


% 将预测概率转换为类别
[~, YPred] = max(YPredProb, [], 2); % 获取概率最高的类别
YPred = categorical(YPred-1); % 偏移索引以匹配类别标签

% Step 8: 评估性能
% 计算准确率
accuracy = sum(YPred == yValSeq) / numel(yValSeq);
fprintf('验证集上的准确率: %.4f±%.4f \n', mean(accuracy), std(accuracy));

% 绘制混淆矩阵
figure;
confusionchart(yValSeq, YPred); 
title('Validation Confusion Matrix');

% 绘制训练集混淆矩阵
YPredTrainProb = predict(net, XTrain, 'MiniBatchSize', 1); % 训练集预测概率
[~, YPredTrain] = max(YPredTrainProb, [], 2); % 获取概率最高的类别
YPredTrain = categorical(YPredTrain-1);
figure;
confusionchart(yTrainSeq, YPredTrain); 
title('Train Confusion Matrix');

%%
figure;
for ii = 1:30
    if ySeq(ii) == categorical(0)
        nexttile;
        imagesc(XSeq{ii});
    end
end
title(0);
figure;
for ii = 1:30%length(XSeq)
    if ySeq(ii) == categorical(1)
        nexttile;
        imagesc(XSeq{ii});
    end
end
title(1)
