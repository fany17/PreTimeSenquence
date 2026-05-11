% MATLAB LSTM Model Training and Evaluation
clear; close all;
% Step 1: Load and Prepare Data
% Step 1: Load and Prepare Data
[X, y] = load_and_prepare_data('LSTMdata.mat');

% 数据集信息
% 时间步数: 570182
% X: 570182x13 double
% y: 570182x1 double (二分类指标，包含0和1)

% Step 2: 设置窗口大小，划分数据为序列段
windowSize = 100; % 定义窗口大小
seglength = 480;
numSegments = floor((size(X, 1) - seglength) / windowSize); % 计算序列段数量

XSeq = {};
ySeq = [];

% 使用滑动窗口创建序列段，每个cell包含一个序列段
for i = 1:numSegments
    startIdx = (i - 1) * windowSize + 1;
    endIdx = startIdx + seglength - 1;
%     XSeq{end+1} = mapminmax(X(startIdx:endIdx, :)',0,1);
    XSeq{end+1} = X(startIdx:endIdx, :)';
    ySeq(end+1) = (y(endIdx)); % 每段的标签为最后一个时间步的标签
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

% Step 4: 定义LSTM网络架构（用于二分类）
numFeatures = size(X,2);
numHiddenUnits = 300; % LSTM层的单元数

layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1') % 一维卷积层
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.5, 'Name', 'dropout1')
%     lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'lstm0')
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'lstm1')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm2')
%     fullyConnectedLayer(128, 'Name', 'fc1')
%     reluLayer('Name', 'relu3')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(2, 'Name', 'fc3') % 输出节点数设为2
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 800, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.001, ... % 增大学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 150, ... % 提高学习率下降频率
    'LearnRateDropFactor', 0.75, ...
    'Verbose', 0, ...
    'Plots', 'training-progress', ...
    'ValidationData', {XValSeq, yValSeq}, ...
    'ValidationFrequency', floor(numel(XTrainSeq)/10));



% % change2 2D
% % % Step 2: 设置窗口大小，划分数据为序列段
% % windowSize = 10;
% % seglength = 480;
% % numSegments = floor((size(X, 1) - seglength) / windowSize);
% % 
% % XSeq = {};
% % ySeq = [];
% % 
% % for i = 1:numSegments
% %     startIdx = (i - 1) * windowSize + 1;
% %     endIdx = startIdx + seglength - 1;
% %     XSegment = X(startIdx:endIdx, :);
% %     XSeq{end+1} = reshape(XSegment, [seglength, numFeatures, 1]); % 转换为2D张量
% %     ySeq(end+1) = y(endIdx);
% % end
% % ySeq = categorical(ySeq);
% % 
% % % Step 3: 随机划分训练集和验证集
% % trainRatio = 0.8;
% % numTrain = floor(trainRatio * numSegments);
% % 
% % randomIdx = randperm(numSegments);
% % trainIdx = randomIdx(1:numTrain);
% % valIdx = randomIdx(numTrain+1:end);
% % 
% % XTrainSeq = XSeq(trainIdx);
% % yTrainSeq = ySeq(trainIdx);
% % XValSeq = XSeq(valIdx);
% % yValSeq = ySeq(valIdx);
% % 
% % % Step 4: 定义2D卷积网络架构
% % layers = [
% %     imageInputLayer([seglength, numFeatures, 1], 'Name', 'input') % 输入尺寸：480x13x1
% %     convolution2dLayer([3, 3], 64, 'Padding', 'same', 'Name', 'conv1') % 2D卷积
% %     batchNormalizationLayer('Name', 'batchnorm1')
% %     reluLayer('Name', 'relu1')
% %     maxPooling2dLayer([2, 2], 'Stride', [2, 2], 'Name', 'pool1')
% %     dropoutLayer(0.5, 'Name', 'dropout1')
% % 
% %     convolution2dLayer([3, 3], 128, 'Padding', 'same', 'Name', 'conv2') % 第二层2D卷积
% %     batchNormalizationLayer('Name', 'batchnorm2')
% %     reluLayer('Name', 'relu2')
% %     maxPooling2dLayer([2, 2], 'Stride', [2, 2], 'Name', 'pool2')
% %     dropoutLayer(0.5, 'Name', 'dropout2')
% % 
% %     fullyConnectedLayer(64, 'Name', 'fc1')
% %     reluLayer('Name', 'relu3')
% %     fullyConnectedLayer(2, 'Name', 'fc2')
% %     softmaxLayer('Name', 'softmax')
% %     classificationLayer('Name', 'output')
% % ];
% % 
% % options = trainingOptions('adam', ...
% %     'MaxEpochs', 250, ...
% %     'GradientThreshold', 1, ...
% %     'InitialLearnRate', 0.0001, ...
% %     'LearnRateSchedule', 'piecewise', ...
% %     'LearnRateDropPeriod', 20, ...
% %     'LearnRateDropFactor', 0.5, ...
% %     'Verbose', 0, ...
% %     'Plots', 'training-progress', ...
% %     'ValidationData', {XValSeq, yValSeq}, ...
% %     'ValidationFrequency', floor(numel(XTrainSeq)/5));
% % 
% % % Step 6: 训练网络
% % net = trainNetwork(XTrainSeq, yTrainSeq, layers, options);


% Step 6: 训练网络
net = trainNetwork(XTrainSeq, yTrainSeq, layers, options);

%%
% acc = testnet(net,XValSeq,yValSeq,"accuracy");
% scores = minibatchpredict(net,XValSeq);
% YPred = scores2label(scores,{'Up','Down'});


% Step 7: 进行预测
YPredProb = predict(net, XValSeq, 'MiniBatchSize', 1); % 预测概率

% 将预测概率转换为类别
[~, YPred] = max(YPredProb, [], 2); % 获取概率最高的类别
% YPred = categorical(YPred-1, [0,1], {'class0','class1'}); % 偏移索引以匹配类别标签
YPred = categorical(YPred-1); % 偏移索引以匹配类别标签

% Step 8: 评估性能
% 计算准确率
accuracy = sum(YPred == yValSeq) / numel(yValSeq);
fprintf('验证集上的准确率: %.4f±%.4f \n', mean(accuracy), std(accuracy));

%%
figure;
confusionchart(yValSeq,YPred); 
title('Validation Confuse Matrix');

%%

YPredTrian = predict(net, XTrainSeq, 'MiniBatchSize', 1); % 预测概率
[~, YPredTrian] = max(YPredTrian, [], 2); % 获取概率最高的类别
YPredTrian = categorical(YPredTrian-1); % 偏移索引以匹配类别标签
figure;
confusionchart(yTrainSeq,YPredTrian); 
title('Train Confuse Matrix');


% % 计算混淆矩阵
% confMat = confusionmat(yValSeq, YPred);
% disp('混淆矩阵:');
% disp(confMat);
% 
% % 可选：计算其他指标如精确率、召回率和F1分数
% if size(confMat,1) == 2 && size(confMat,2) == 2
%     precision = confMat(2,2) / sum(confMat(:,2));
%     recall = confMat(2,2) / sum(confMat(2,:));
%     f1Score = 2 * (precision * recall) / (precision + recall);
%     fprintf('精确率: %.4f\n', precision);
%     fprintf('召回率: %.4f\n', recall);
%     fprintf('F1分数: %.4f\n', f1Score);
% end

%%
% 假设 YPredProb 是预测的概率
class0_probs = YPredProb(double(yValSeq)-1 == 0, 2); % Class 0 的预测概率
class1_probs = YPredProb(double(yValSeq)-1  == 1, 2); % Class 1 的预测概率

% 绘制概率分布
figure;
histogram(class0_probs, 'Normalization', 'pdf', 'DisplayName', 'Class 0');
hold on;
histogram(class1_probs, 'Normalization', 'pdf', 'DisplayName', 'Class 1');
hold off;
xlabel('Predicted Probability');
ylabel('Density');
legend;
title('Prediction Probability Distribution');



%%
% 使用 TSNE 降维
% Step 7: 进行预测并提取特征嵌入
featuresVal = activations(net, XValSeq, 'lstm2', 'OutputAs', 'rows');

% 将预测概率转换为类别
[~, YPred] = max(YPredProb, [], 2); % 获取概率最高的类别
YPred = categorical(YPred-1, [0,1], {'class0','class1'}); % 偏移索引以匹配类别标签

% 使用 TSNE 降维
Y_embedded = tsne(featuresVal);

% 绘制 TSNE 降维可视化（使用真实标签和预测标签）
figure;
subplot(1,2,1);
gscatter(Y_embedded(:,1), Y_embedded(:,2), yValSeq); % 使用真实标签
title('TSNE Visualization with True Labels');
xlabel('Dimension 1');
ylabel('Dimension 2');

subplot(1,2,2);
gscatter(Y_embedded(:,1), Y_embedded(:,2), YPred); % 使用预测标签
title('TSNE Visualization with Predicted Labels');
xlabel('Dimension 1');
ylabel('Dimension 2');


%%
function [X, y] = load_and_prepare_data(file_path)
    % 加载 .mat 文件并转换为结构体
    data = load(file_path);
    df = data;  % 假设文件加载后的数据直接是结构体

    % 计算对数收益率并作为新特征
    df.log_return = log(df.close ./ [NaN df.close(1:end-1)]);
    % 计算移动平均线（5日和20日）
    df.MA5 = movmean(df.close, 5);
    df.MA20 = movmean(df.close, 20);
    
    % 计算RSI（14日）
    df.RSI = 100 - (100 ./ (1 + movmean(max(0, df.close - [NaN df.close(1:end-1)]), 14) ./ ...
                           movmean(abs(df.close - [NaN  df.close(1:end-1)]), 14)));
    
    % 计算布林带
    std_20 = movstd(df.close, 20);
    df.BB_upper = df.MA20 + 2 * std_20;
    df.BB_lower = df.MA20 - 2 * std_20;
    
    % 计算5日、10日的价格变化率
    df.price_change_5 = (df.close - [NaN(5,1)'  df.close(1:end-5)]) ./ [NaN(5,1)' df.close(1:end-5)];
    df.price_change_10 = (df.close - [NaN(10,1)'  df.close(1:end-10)]) ./ [NaN(10,1)' df.close(1:end-10)];
    
    % 提取所有特征并进行归一化
    features = {'open', 'high', 'low', 'close', 'volume', 'log_return', 'MA5', 'MA20', 'RSI', ...
                'BB_upper', 'BB_lower', 'price_change_5', 'price_change_10'};
    

    % 将 position_signal 映射为 long=1, short=0, none=NaN
    df.position_signal = cellfun(@(x) strcmp(x, 'long') * 1 + strcmp(x, 'short') * 0, df.position_signal, 'UniformOutput', true);

    % 提取数值特征矩阵并进行归一化
    X = cell2mat(cellfun(@(f) df.(f)', features, 'UniformOutput', false))';
%     X = mapminmax(X', 0, 1);
    
    % 提取标签
    y = df.position_signal;

    % 删除包含 NaN 的列
    nan_columns = any(isnan([X; y]), 1);  % 找到 X 或 y 中包含 NaN 的列
    X(:, nan_columns) = [];  % 去除 X 中包含 NaN 的列
    y(nan_columns) = [];     % 去除 y 中对应的列
    X = X';
    y = y';
end