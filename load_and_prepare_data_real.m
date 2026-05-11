function X = load_and_prepare_data_real(df)
    %% 基本特征计算
    % 对数收益率
    df.log_return = log(df.close ./ [NaN df.close(1:end-1)]);
    
    % 移动平均线
    df.MA5 = movmean(df.close, [5 0]);
    df.MA20 = movmean(df.close, [20 0]);
    df.MA50 = movmean(df.close, [50 0]);
    df.MA100 = movmean(df.close, [100 0]);
    
%     % 波动率与布林带
%     std_20 = movstd(df.close, 20);
%     df.BB_upper = df.MA20 + 2 * std_20;
%     df.BB_lower = df.MA20 - 2 * std_20;
%     df.BB_width = df.BB_upper - df.BB_lower;

    periods = [5, 10, 20, 50, 100];
%     periods = 20;
    for p = periods
        std_p = movstd(df.close, [p-1 0]);
        df.(['BB_upper_' num2str(p)]) = movmean(df.close, [p-1 0]) + 2 * std_p;
        df.(['BB_lower_' num2str(p)]) = movmean(df.close, [p-1 0]) - 2 * std_p;
        df.(['BB_width_' num2str(p)]) = df.(['BB_upper_' num2str(p)]) - df.(['BB_lower_' num2str(p)]);
    end

    
    %% 动量和趋势特征
    % 动量
    df.momentum_5 = df.close - [NaN(1,5) df.close(1:end-5)];
    df.momentum_20 = df.close - [NaN(1,20) df.close(1:end-20)];
    df.momentum_10 = df.close - [NaN(1,10) df.close(1:end-10)];
    
    % 价格变化率 (Rate of Change)
    df.roc_5 = (df.close - [NaN(1,5) df.close(1:end-5)]) ./ [NaN(1,5) df.close(1:end-5)];
    df.roc_10 = (df.close - [NaN(1,10) df.close(1:end-10)]) ./ [NaN(1,10) df.close(1:end-10)];
    df.roc_20 = (df.close - [NaN(1,20) df.close(1:end-20)]) ./ [NaN(1,20) df.close(1:end-20)];
    df.roc_30 = (df.close - [NaN(1,30) df.close(1:end-30)]) ./ [NaN(1,30) df.close(1:end-30)];
    df.roc_60 = (df.close - [NaN(1,60) df.close(1:end-60)]) ./ [NaN(1,60) df.close(1:end-60)];
    
    % True Strength Index (TSI)
    df.TSI = movmean(movmean(df.log_return, [25 0]), [13 0]);
    
    %% 高级技术指标
    % MACD Histogram
    df.EMA12 = movmean(df.close, [12 0]);
    df.EMA26 = movmean(df.close, [26 0]);
    df.MACD = df.EMA12 - df.EMA26;
    df.MACD_signal = movmean(df.MACD, [9 0]);
    df.MACD_hist = df.MACD - df.MACD_signal;
    
    % Stochastic Oscillator
    highest_high = movmax(df.high, [14 0]);
    lowest_low = movmin(df.low, [14 0]);
    df.stochastic_k = (df.close - lowest_low) ./ (highest_high - lowest_low) * 100;
    df.stochastic_d = movmean(df.stochastic_k, [3 0]);
    
    % Ichimoku Cloud Components
    % Tenkan-sen (9-period)
    df.Tenkan_sen = (movmax(df.high, [9 0]) + movmin(df.low, [9 0])) / 2;
    % Kijun-sen (26-period)
    df.Kijun_sen = (movmax(df.high, [26 0]) + movmin(df.low, [26 0])) / 2;
    % Senkou Span A (leading span A)
    df.Senkou_Span_A = (df.Tenkan_sen + df.Kijun_sen) / 2;
    % Senkou Span B (leading span B, 52-period)
    df.Senkou_Span_B = (movmax(df.high, [52 0]) + movmin(df.low, [52 0])) / 2;
%     % Chikou Span (lagging span)
%     df.Chikou_Span = [df.close(21:end)  NaN(1,20)];
    
    % Parabolic SAR
    df.PSAR = parabolicSAR(df.high, df.low, 0.02, 0.2);
    
    % Average Directional Index (ADX)
    df.positive_DM = max(df.high - [NaN df.high(1:end-1)], 0);
    df.negative_DM = max([NaN df.low(1:end-1)] - df.low, 0);
    df.TR = max([df.high - df.low, abs(df.high - [NaN df.close(1:end-1)]), abs(df.low - [NaN df.close(1:end-1)])], [], 2);
    df.PDI = 100 * movmean(df.positive_DM, [14 0]) ./ movmean(df.TR, [14 0]);
    df.NDI = 100 * movmean(df.negative_DM, [14 0]) ./ movmean(df.TR, [14 0]);
    df.DX = 100 * abs(df.PDI - df.NDI) ./ (df.PDI + df.NDI);
    df.ADX = movmean(df.DX, [14 0]);
    
    %% 成交量特征
    df.Volume_MA5 = movmean(df.volume, [5 0]);
    df.Volume_MA20 = movmean(df.volume, [20 0]);
    df.price_to_volume = df.close ./ (df.volume + eps);
    df.volume_acceleration = [NaN NaN diff(diff(df.volume))];
    
    % On-Balance Volume (OBV)
    df.OBV = zeros(size(df.close));
    df.OBV(1) = df.volume(1);
    for i = 2:length(df.close)
        if df.close(i) > df.close(i-1)
            df.OBV(i) = df.OBV(i-1) + df.volume(i);
        elseif df.close(i) < df.close(i-1)
            df.OBV(i) = df.OBV(i-1) - df.volume(i);
        else
            df.OBV(i) = df.OBV(i-1);
        end
    end
    df.OBV_change = [NaN  diff(df.OBV)];
    
    % Money Flow Index (MFI)
    typical_price = (df.high + df.low + df.close) / 3;
    df.MF = typical_price .* df.volume;
    df.MF_positive = zeros(size(df.MF));
    df.MF_negative = zeros(size(df.MF));
    for i = 2:length(df.MF)
        if df.MF(i) > df.MF(i-1)
            df.MF_positive(i) = df.MF(i);
            df.MF_negative(i) = 0;
        elseif df.MF(i) < df.MF(i-1)
            df.MF_positive(i) = 0;
            df.MF_negative(i) = df.MF(i);
        else
            df.MF_positive(i) = 0;
            df.MF_negative(i) = 0;
        end
    end
    df.MFI_positive_sum = movsum(df.MF_positive, [14 0]);
    df.MFI_negative_sum = movsum(df.MF_negative, [14 0]);
    df.MFI = 100 - (100 ./ (1 + df.MFI_positive_sum ./ (df.MFI_negative_sum + eps)));
    
    %% 统计特征
%     df.rolling_skew = movskew(df.log_return, 20);
        % 计算 20 日滚动偏度 (skewness)
    window_size = 20;
    df.rolling_skew = NaN(size(df.log_return)); % 预分配结果数组
    for i = window_size:length(df.log_return)
        df.rolling_skew(i) = skewness(df.log_return(i-window_size+1:i));
    end

%     df.rolling_kurtosis = movkurtosis(df.log_return, 20);
        % 计算 20 日滚动峰度 (kurtosis)
    df.rolling_kurtosis = NaN(size(df.log_return)); % 预分配结果数组
    for i = window_size:length(df.log_return)
        df.rolling_kurtosis(i) = kurtosis(df.log_return(i-window_size+1:i));
    end

    
    % Hurst Exponent (简化版计算)
    df.Hurst = zeros(size(df.close));
    window = 100;
    for i = window:length(df.close)
        series = df.log_return(i-window+1:i);
        N = length(series);
        mean_val = mean(series);
        Y = cumsum(series - mean_val);
        R = max(Y) - min(Y);
        S = std(series);
        if S > 0
            df.Hurst(i) = log(R/S) / log(N);
        else
            df.Hurst(i) = NaN;
        end
    end
    
%     %% 时间特征
%     % 假设df有日期字段，格式为datetime
%     if isfield(df, 'date')
%         df.day_of_week = weekday(df.date); % 1=Sunday, 2=Monday, ..., 7=Saturday
%         df.is_month_start = ismonthstart(df.date);
%         df.is_month_end = ismonthend(df.date);
%     else
%         % 如果没有日期字段，可以跳过或使用其他时间特征
%         df.day_of_week = NaN(size(df.close));
%         df.is_month_start = NaN(size(df.close));
%         df.is_month_end = NaN(size(df.close));
%     end
%     
    %% 价格模式识别 (示例：Hammer Pattern)
    df.is_hammer = zeros(size(df.close));
    for i = 2:length(df.close)
        body = abs(df.close(i) - df.open(i));
        range = df.high(i) - df.low(i);
        lower_shadow = min(df.close(i), df.open(i)) - df.low(i);
        if (lower_shadow > 2 * body) && (body / range < 0.3)
            df.is_hammer(i) = 1;
        else
            df.is_hammer(i) = 0;
        end
    end
    
    %% Lagged Features
    % 滞后特征 (过去5天的MA20)
    for lag = 1:10
        df.(['MA20_lag_' num2str(lag)]) = [NaN(1,lag)  df.MA20(1:end-lag)];
    end
    %%
    period = 14;
    df.up = max(df.close - [NaN df.close(1:end-1)], 0);
    df.down = max([NaN df.close(1:end-1)] - df.close, 0);
    df.avg_up = movmean(df.up, [period-1 0]);
    df.avg_down = movmean(df.down, [period-1 0]);
    df.RSI = 100 - 100 ./ (1 + df.avg_up ./ (df.avg_down + eps));
    
    %% 提取所有特征
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

    for p = periods
        features{end+1} = ['BB_upper_' num2str(p)];
        features{end+1} = ['BB_lower_' num2str(p)];
        features{end+1} = ['BB_width_' num2str(p)];
    end
    
    % 添加滞后特征
    for lag = 1:10
        features{end+1} = ['MA20_lag_' num2str(lag)];
    end
    
    %% 将 position_signal 映射为 long=1, short=0
    
     % 提取数值特征矩阵并进行归一化
    X = cell2mat(cellfun(@(f) df.(f)', features, 'UniformOutput', false))';
%     X = mapminmax(X,0,1)';

    X = X';
end

%% 辅助函数：Parabolic SAR 计算
function psar = parabolicSAR(high, low, acc_init, acc_max)
    % 初始化
    psar = NaN(size(high));
    length_data = length(high);
    if length_data < 1
        return;
    end
    % 初始点
    psar(1) = low(1);
    trend = 1; % 1: 上升, -1: 下降
    acc = acc_init;
    ep = high(1);
    
    for i = 2:length_data
        % 计算当前的SAR
        if trend == 1
            psar(i) = psar(i-1) + acc * (ep - psar(i-1));
        else
            psar(i) = psar(i-1) - acc * (psar(i-1) - ep);
        end
        
        % 反转条件
        if trend == 1
            if low(i) < psar(i)
                trend = -1;
                psar(i) = ep;
                ep = low(i);
                acc = acc_init;
            else
                % 更新 EP
                if high(i) > ep
                    ep = high(i);
                    acc = min(acc + acc_init, acc_max);
                end
            end
        else
            if high(i) > psar(i)
                trend = 1;
                psar(i) = ep;
                ep = high(i);
                acc = acc_init;
            else
                % 更新 EP
                if low(i) < ep
                    ep = low(i);
                    acc = min(acc + acc_init, acc_max);
                end
            end
        end
    end
end
