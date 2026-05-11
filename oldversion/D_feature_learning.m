clear; close all;

% Step 1: 加载数据
[X, y] = load_and_prepare_data('LSTMdata.mat');

%%

% TODO: 重写y的设计。
y0 = y;
d = [0;diff(X(:,4))];
y0(d>0) = 1;
y0(d<0) = 0;

%%
t = 1:length(X);
z = y0;
figure;tiledlayout(2,1);
ax1=nexttile;
plot(X(:,4));hold on;
plot(t(z==1),X(z==1,4),'.r');
plot(t(z==0),X(z==0,4),'.g');
title('Pred');
ax2=nexttile;
plot(X(:,4));hold on;
plot(t(y==1),X(y==1,4),'.r');
plot(t(y==0),X(y==0,4),'.g');
title('True');
linkaxes([ax1, ax2], 'xy'); % 'xy' 表示同时同步 X 和 Y 坐标

%%

figure;
confusionchart(y, y0);
accuracy = sum(y == y0) / numel(y0);
title([num2str(mean(accuracy) * 100),'%'])

%% 增加盈利比较


%%
features = {
    'open', 'high', 'low', 'close', 'volume', ...
    'MA5', 'MA20', 'MA50', 'MA100', ...
    'log_return', 'momentum_5', 'momentum_10', 'momentum_20', ...
    'roc_5', 'roc_10', 'roc_20', 'roc_30', 'roc_60', ...
    'TSI', ...
    'MACD', 'MACD_signal', 'MACD_hist', ...
    'stochastic_k', 'stochastic_d', ...
    'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', ...
    'PSAR', ...
    'ADX', 'PDI', 'NDI', 'DX', ...
    'Volume_MA5', 'Volume_MA20', 'price_to_volume', 'volume_acceleration', ...
    'OBV', 'OBV_change', 'MFI', ...
    'rolling_skew', 'rolling_kurtosis',...
    'is_hammer',...
    'up','down','avg_up','avg_down','RSI'
};

periods = [5, 10, 20, 50, 100];
for p = periods
    features{end+1} = ['BB_upper_' num2str(p)];
    features{end+1} = ['BB_lower_' num2str(p)];
    features{end+1} = ['BB_width_' num2str(p)];
end

% 添加滞后特征
for lag = 1:10
    features{end+1} = ['MA20_lag_' num2str(lag)];
end
%%
t = 1:length(y);
long = y==1;
short = y==0;
figure;hold on;
testingfeature = X(:,2)-X(:,55);
featuresname = [features{2},'-',features{55}];
plot(testingfeature,'-');
plot(t(long),testingfeature(long),'.r',DisplayName='long');
plot(t(short),testingfeature(short),'.g',DisplayName='short');
legend;
title(featuresname,'Interpreter','none');



%%
% Step 3: 特征相关性分析
% 使用皮尔逊相关系数计算各特征之间的相关性
correlationMatrix = corr(X);
figure;
heatmap(correlationMatrix);
title('Feature Correlation Matrix');
xlabel('Features');
ylabel('Features');

% Step 4: 特征与标签的相关性
% 使用信息增益或Fisher评分等方法，计算各特征与标签的相关性
featureCorrelation = corr(X, y, 'type', 'Spearman');
figure;
bar(featureCorrelation);
title('Feature-Label Correlation');
xlabel('Features');
ylabel('Spearman Correlation');

% Step 5: 特征分布可视化
% 使用直方图或箱线图观察特征的分布
numFeatures = size(X, 2);
figure;
for i = 1:min(numFeatures, 10) % 绘制前10个特征
    subplot(5, 2, i);
    histogram(X(:, i));
    title(['Feature ', num2str(i), ' Distribution']);
end

% Step 6: 使用PCA降维进行特征可视化
% PCA可用于观察数据的主成分分布，帮助我们了解特征的有效性
[coeff, score, ~] = pca(X);
figure;
gscatter(score(:, 1), score(:, 2), y, 'br', 'ox');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA Feature Visualization');
legend('Class 0', 'Class 1');