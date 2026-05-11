% % % % 清理环境
clear; close all; clc;

% Step 1: 加载数据并降维
[X, y0] = load_and_prepare_data('LSTMdata.mat');
%%
Xn = X;
y = y0;
% parfor i = 100:size(X,1)
%     disp(i);
%     Xtemp = mapminmax(X(i-99:i,:)')';
%     Xn(i,:) = Xtemp(end,:);
% end
% Xn = Xn(100:end,:);
% y = y0(100:end);
% normlist = round(log10(max(abs(X))));
% for i = 1:size(X,2)
%     Xn(:,i) = X(:,i)./10^(normlist(i));
% end
imagesc(Xn);
% save('normlist.mat',"normlist");
%%

% Step 2: 分离训练和测试数据
cv = cvpartition(size(y,1), 'HoldOut', 0.6);
XTrain = Xn(training(cv), :);
yTrain = y(training(cv), :);
XTest = Xn(test(cv), :);
yTest = y(test(cv), :);



% % % Step 3: 使用SVM进行训练
% SVMModel = fitcsvm(XTrain, yTrain, 'KernelFunction', 'linear', 'ClassNames', [0, 1]);
% % 使用 fitclinear 训练线性 SVM
SVMModel = fitclinear(XTrain, yTrain, 'Learner', 'svm');


%%
% Step 5: 在测试数据上评估
[yPredraw, score] = predict(SVMModel, XTest);
accuracy = sum(yPredraw == yTest) / numel(yTest);
fprintf('SVM测试集原始分类准确率: %.2f%%\n', mean(accuracy) * 100);

% 可视化混淆矩阵
figure;
confusionchart(yTest, yPredraw);
title('SVM 分类器原始混淆矩阵');

%%
% Step 4: 在边界范围内的样本定义为不确定区域

epsilon = 0.1; % 设置分类边界宽度
yPred = score2pred(score,epsilon);
yPredO = yPred;
yTestO = yTest;
yPredO(yPred==0.5)=[];
yTestO(yPred==0.5)=[];

accuracy = sum(yPredO == yTestO) / numel(yTestO);
fprintf('SVM测试集修改后分类准确率: %.2f%%\n', mean(accuracy) * 100);

figure;
confusionchart(yTestO, yPredO);
title('SVM 分类器修改后混淆矩阵');

%%
Xt = X(100:end,:);
[z, ~] = predict(SVMModel, Xn);
t = 1:length(Xt);
figure;tiledlayout(2,1);
ax1=nexttile;
plot(Xt(:,4));hold on;
plot(t(z==1),Xt(z==1,4),'.r');
plot(t(z==0),Xt(z==0,4),'.g');
title('Pred');
ax2=nexttile;
plot(Xt(:,4));hold on;
plot(t(y==1),Xt(y==1,4),'.r');
plot(t(y==0),Xt(y==0,4),'.g');
title('True');
linkaxes([ax1, ax2], 'xy'); % 'xy' 表示同时同步 X 和 Y 坐标

%%
[~, zs] = predict(SVMModel, Xn);
z = score2pred(zs,0.3);
t = 1:length(Xt);
figure;tiledlayout(2,1);
ax1=nexttile;
plot(Xt(:,4));hold on;
plot(t(z==1),Xt(z==1,4),'.r');
plot(t(z==0),Xt(z==0,4),'.g');
% plot(t(z==0.5),Xn(z==0.5,4),'.k');
title('Pred');
ax2=nexttile;
plot(Xt(:,4));hold on;
plot(t(y==1),Xt(y==1,4),'.r');
plot(t(y==0),Xt(y==0,4),'.g');
title('True');
linkaxes([ax1, ax2], 'xy'); % 'xy' 表示同时同步 X 和 Y 坐标


%% Saveing

SVMModel = fitclinear(Xn, y, 'Learner', 'svm');
save('SVMmodel.mat',"SVMModel");


%%

function yPred = score2pred(score,epsilon)
score = score(:,2);
probabilities = 1 ./ (1 + exp(-score)); % 将分数映射到 [0, 1]
% epsilon = 0.1; % 设置分类边界宽度
% 定义预测结果（允许边界范围内的输出）
yPred = probabilities;
yPred(probabilities > 0.5 + epsilon) = 1; % 高于边界范围归为 1 类
yPred(probabilities < 0.5 - epsilon) = 0; % 低于边界范围归为 0 类
yPred(abs(probabilities - 0.5) <= epsilon) = 0.5; % 边界范围内定义为不确定
end


