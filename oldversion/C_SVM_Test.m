% 清理环境
clear; close all; clc;

% Step 1: 加载数据并降维
[X, y] = load_and_prepare_data('LSTMdata_test.mat');

%%
Xn = X;
% load('normlist.mat',"normlist");
% for i = 1:size(X,2)
%     Xn(:,i) = X(:,i)./10^(normlist(i));
% end
% imagesc(Xn);

load('SVMmodel.mat',"SVMModel");

[~, score] = predict(SVMModel, Xn);

epsilon = 0.45; % 设置分类边界宽度
yPred = score2pred(score,epsilon);
yPredO = yPred;
yTestO = y;
yPredO(yPred==0.5)=[];
yTestO(yPred==0.5)=[];

accuracy = sum(yPredO == yTestO) / numel(yTestO);
fprintf('SVM测试集修改后分类准确率: %.2f%%\n', mean(accuracy) * 100);

figure;
confusionchart(yTestO, yPredO);
title('SVM 分类器修改后混淆矩阵');

z = score2pred(score,epsilon);
t = 1:length(Xn);
figure;tiledlayout(2,1);
ax1=nexttile;
plot(Xn(:,4));hold on;
plot(t(z==1),Xn(z==1,4),'.r');
plot(t(z==0),Xn(z==0,4),'.g');
% plot(t(z==0.5),Xn(z==0.5,4),'.k');
title('Pred');
ax2=nexttile;
plot(Xn(:,4));hold on;
plot(t(y==1),Xn(y==1,4),'.r');
plot(t(y==0),Xn(y==0,4),'.g');
title('True');
linkaxes([ax1, ax2], 'xy'); % 'xy' 表示同时同步 X 和 Y 坐标


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
