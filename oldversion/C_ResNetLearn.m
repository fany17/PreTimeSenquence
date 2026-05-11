% MATLAB CNN Model Training and Evaluation with ResNet-50
clear; close all; clc;

% Step 1: Load and Prepare Data
[X, y] = load_and_prepare_data('LSTMdata.mat');

% 数据集信息
% 时间步数: 570182
% X: 570182x13 double
% y: 570182x1 double (二分类指标，包含0和1)
%%
% Step 2: 设置窗口大小，划分数据为序列段
windowSize = 40; % 定义窗口大小
seglength = 53;
numSegments = floor((size(X, 1) - seglength) / windowSize); % 计算序列段数量

XSeq = {};
ySeq = [];

% 使用滑动窗口创建序列段，每个cell包含一个序列段
for i = 1:numSegments
    startIdx = (i - 1) * windowSize + 1;
    endIdx = startIdx + seglength - 1;
    XSegment = X(startIdx:endIdx, :);
    XSegment = mapminmax(XSegment', 0, 1)';    
    % 调整为 224x224x3
    XResized = imresize(XSegment, [224, 224]); % 调整大小
    XResized = repmat(XResized, [1, 1, 3]);    % 扩展到3通道
    XSeq{end+1} = XResized; % 转换为3D张量
    ySeq(end+1) = y(endIdx); % 每段的标签为最后一个时间步的标签
end
ySeq = categorical(ySeq);

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

%%
% Step 4: 加载 ResNet-50 并修改最后的层
% 加载 ResNet-18 代替 ResNet-50
% net = resnet18;
% % 转换为 layerGraph 对象
% lgraph = layerGraph(net);
% numClasses = 2;
% newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
%     'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
% newClassificationLayer = classificationLayer('Name', 'output');
% lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer); % ResNet-18 最后的全连接层仍然命名为 'fc1000'
% lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassificationLayer);


% net = squeezenet;
% % 替换最后几层用于二分类
% lgraph = layerGraph(net);
% numClasses = 2;
% newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
%     'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
% newClassificationLayer = classificationLayer('Name', 'output');
% lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassificationLayer);
% lgraph = replaceLayer(lgraph, 'conv10', newFCLayer);

net = mobilenetv2; % 加载预训练的 MobileNetV2 模型
lgraph = layerGraph(net);
numClasses = 2; % 根据您的分类任务设置
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'Logits', newFCLayer);
newClassificationLayer = classificationLayer('Name', 'new_output');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newClassificationLayer);


% 定义训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {XVal, yValSeq}, ...
    'ValidationFrequency', floor(numTrain / 2), ...
    'Verbose', false, ...
    'Plots', 'training-progress');


% options = trainingOptions('adam', ...
%     'MiniBatchSize', 32, ...
%     'MaxEpochs', 5, ...
%     'InitialLearnRate', 1e-4, ...
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');


plot(lgraph);

%%
% 训练网络
netTransfer = trainNetwork(XTrain, yTrainSeq, lgraph, options);


% Step 7: 进行预测
YPred = classify(netTransfer, XVal, 'MiniBatchSize', 32);

% Step 8: 评估性能
% 计算准确率
accuracy = sum(YPred == yValSeq) / numel(yValSeq);
fprintf('验证集上的准确率: %.2f%%\n', mean(accuracy) * 100);

% 绘制混淆矩阵
figure;
confusionchart(yValSeq, YPred); 
title('Validation Confusion Matrix');

% 可选：绘制训练集混淆矩阵
YPredTrain = classify(netTransfer, XTrain, 'MiniBatchSize', 32);
figure;
confusionchart(yTrainSeq, YPredTrain); 
title('Train Confusion Matrix');
