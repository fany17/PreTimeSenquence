% 清理环境
clear; close all; clc;

% Step 1: 加载数据并降维
%%

data = py.AGetData4.get_latest_data;
df = struct( ...
    'open', double(data{'open'}), ...
    'high', double(data{'high'}), ...
    'low', double(data{'low'}), ...
    'close', double(data{'close'}), ...
    'volume', double(data{'volume'}) ...
);


%%
X = load_and_prepare_data_real(df);

%%
Xn = X;
load('normlist.mat',"normlist");
for i = 1:size(X,2)
    Xn(:,i) = X(:,i)./10^(normlist(i));
end
% imagesc(Xn);

load('SVMmodel.mat',"SVMModel");

[~, score] = predict(SVMModel, Xn);

epsilon = 0.45; % 设置分类边界宽度
yPred = score2pred(score,epsilon);

disp(yPred(end));


z = score2pred(score,epsilon);
t = 1:length(Xn);
figure('Visible','off');
set(gcf,'Position',[847,-667,527,166])
% set(gcf,'Position',[825,-737,560,420])
% set(gcf,'Position',[825,437,560,420])

% ax1=nexttile;
plot(Xn(:,4),'.-',LineWidth=1.5);hold on;
plot(t(z==1),Xn(z==1,4),'.','MarkerSize',20,color="#cf292f");
plot(t(z==0),Xn(z==0,4),'.','MarkerSize',20,color="#006e5f");
% plot(t(z==0.5),Xn(z==0.5,4),'.k');

if yPred(end) == 1
    positionpred="Long";
elseif yPred(end) == 0
    positionpred="short";
else
    positionpred="None";
end
xlim([60 210]);axis off;
title(positionpred,"FontSize",15);

saveas(gcf,"temp.png");
% ax2=nexttile;
% plot(Xn(:,4));hold on;
% plot(t(y==1),Xn(y==1,4),'.r');
% plot(t(y==0),Xn(y==0,4),'.g');
% title('True');
% linkaxes([ax1, ax2], 'xy'); % 'xy' 表示同时同步 X 和 Y 坐标


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
