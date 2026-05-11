from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.svm import SVC
import pandas_ta as ta
import os
import nolds
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta  # 使用 pandas_ta 库


def parabolic_sar(high, low, acceleration=0.02, maximum=0.2):
    """
    计算抛物线转向指标（Parabolic SAR）
    """
    psar = np.zeros_like(high)
    psar[0] = low[0]
    bull = True
    af = acceleration
    ep = high[0]

    for i in range(1, len(high)):
        prev_psar = psar[i - 1]
        if bull:
            psar[i] = prev_psar + af * (ep - prev_psar)
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = acceleration
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
        else:
            psar[i] = prev_psar - af * (prev_psar - ep)
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = acceleration
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)
    return psar


def load_and_prepare_data(file_path):
    """
    加载数据并处理特征，整合 MATLAB 和 Python 的处理逻辑。
    :param file_path: 数据文件路径
    :return: 特征矩阵 X 和目标 y
    """
    # 加载数据
    df = pd.read_pickle(file_path)

    # 必要字段检查
    required_fields = ['open', 'high', 'low', 'close', 'volume', 'position_signal']
    for field in required_fields:
        if field not in df.columns:
            raise ValueError(f"缺少必要字段: {field}")

    # 对数收益率
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 移动均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA100'] = df['close'].rolling(window=100).mean()

    # 布林带
    df['std_20'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * df['std_20']
    df['BB_lower'] = df['MA20'] - 2 * df['std_20']
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    # 动量和趋势特征
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)

    # 价格变化率 (Rate of Change)
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)
    df['roc_20'] = df['close'].pct_change(20)
    df['roc_30'] = df['close'].pct_change(30)
    df['roc_60'] = df['close'].pct_change(60)

    # True Strength Index (TSI)
    tsi = df.ta.tsi(close=df['close'], fast=13, slow=25)
    if isinstance(tsi, pd.DataFrame):
        df['TSI'] = tsi.iloc[:, 0]  # 选择第一列作为结果
    else:
        df['TSI'] = tsi  # 单列结果直接赋值

    # MACD 指标
    macd = df.ta.macd(close=df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']

    # 随机震荡指标 (Stochastic Oscillator)
    stoch = df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=14, d=3, smooth_k=3)
    df['stochastic_k'] = stoch['STOCHk_14_3_3']
    df['stochastic_d'] = stoch['STOCHd_14_3_3']

    # Ichimoku 云指标
    # 计算 Ichimoku 指标
    ichimoku = df.ta.ichimoku(high=df['high'], low=df['low'], close=df['close'], tenkan=9, kijun=26, senkou=52)
    # 合并两个部分
    ichimoku_combined = pd.concat(ichimoku, axis=0)
    # 提取所需的列
    df['Tenkan_sen'] = ichimoku_combined['ITS_9']    # 转换线 (Tenkan-sen)
    df['Kijun_sen'] = ichimoku_combined['IKS_26']    # 基准线 (Kijun-sen)
    df['Senkou_Span_A'] = ichimoku_combined['ISA_9']  # 先行 A 线 (Senkou Span A)
    df['Senkou_Span_B'] = ichimoku_combined['ISB_26']  # 先行 B 线 (Senkou Span B)

    # Parabolic SAR
    df['PSAR'] = parabolic_sar(df['high'].values, df['low'].values)

    # 平均方向性指数 (ADX)
    adx = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['PDI'] = adx['DMP_14']
    df['NDI'] = adx['DMN_14']
    df['DX'] = abs(df['PDI'] - df['NDI']) / (df['PDI'] + df['NDI']) * 100

    # 成交量特征
    df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
    df['price_to_volume'] = df['close'] / (df['volume'] + 1e-9)
    df['volume_acceleration'] = df['volume'].diff().diff()

    # 量价指标
    # On-Balance Volume (OBV)
    df['OBV'] = df.ta.obv(close=df['close'], volume=df['volume'])
    df['OBV_change'] = df['OBV'].diff()

    # Money Flow Index (MFI)
    # 计算 MFI
    df['MFI'] = df.ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=14)
    # 确保列类型为浮点数
    df['MFI'] = df['MFI'].astype(float)

    # 使用优化函数
    df['rolling_skew'] = df.ta.skew(close=df['log_return'], length=20)
    df['rolling_kurtosis'] = df.ta.kurtosis(close=df['log_return'], length=20)

    # # 时间特征
    # if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
    #     df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, ..., 6=Sunday
    #     df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    #     df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    # else:
    #     df['day_of_week'] = np.nan
    #     df['is_month_start'] = np.nan
    #     df['is_month_end'] = np.nan

    # 价格模式识别 (示例：Hammer Pattern)
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    lower_shadow = np.minimum(df['close'], df['open']) - df['low']
    df['is_hammer'] = ((lower_shadow > 2 * body) & (body / range_ < 0.3)).astype(int)

    # 滞后特征 (过去5天的MA20)
    for lag in range(1, 6):
        df[f'MA20_lag_{lag}'] = df['MA20'].shift(lag)

    # 提取所有特征
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'MA5', 'MA20', 'MA50', 'MA100',
        'BB_upper', 'BB_lower', 'BB_width',
        'log_return', 'momentum_5', 'momentum_10', 'momentum_20',
        'roc_5', 'roc_10', 'roc_20', 'roc_30', 'roc_60',
        'TSI',
        'MACD', 'MACD_signal', 'MACD_hist',
        'stochastic_k', 'stochastic_d',
        'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B',
        'PSAR',
        'ADX', 'PDI', 'NDI', 'DX',
        'Volume_MA5', 'Volume_MA20', 'price_to_volume', 'volume_acceleration',
        'OBV', 'OBV_change', 'MFI',
        'rolling_skew', 'rolling_kurtosis',
        'is_hammer'
    ]

    # 添加滞后特征
    for lag in range(1, 6):
        features.append(f'MA20_lag_{lag}')

    # 将 position_signal 映射为 long=1, short=0
    df['position_signal'] = df['position_signal'].map({'long': 1, 'short': 0})

    # 删除包含 NaN 的行
    df.dropna(subset=features + ['position_signal'], inplace=True)

    # 特征和标签
    X = df[features].values
    y = df['position_signal'].values

    save_normlist = 0

    # 保存 normlist 到文件
    if save_normlist:
        # 计算 normlist
        normlist = np.round(np.log10(np.max(np.abs(X), axis=0)))
        np.save("normlist.npy", normlist)
        print("normlist 已保存到 normlist.npy 文件。")
    else:
        normlist = np.load("normlist.npy")

    # 归一化特征矩阵
    Xn_new = X / (10 ** normlist)

    print("数据已归一化。")

    return Xn_new, y


# Step 2: LSTM Model in PyTorch


# class SVMmodels:
#     def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
#         self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

#     def train(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X):
#         return self.model.predict(X)

#     def predict_proba(self, X):
#         return self.model.predict_proba(X)

#     def save_model(self, model_save_path):
#         import joblib
#         joblib.dump(self.model, model_save_path)

#     def load_model(self, model_save_path):
#         import joblib
#         self.model = joblib.load(model_save_path)


class SVMmodels:
    def __init__(self, C=1.0, max_iter=10000):
        self.model = LinearSVC(C=C, max_iter=max_iter, dual=False)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_save_path):
        import joblib
        joblib.dump(self.model, model_save_path)

    def load_model(self, model_save_path):
        import joblib
        self.model = joblib.load(model_save_path)


# Step 3: Train the LSTM Model in PyTorch

def train_svm_model(model, X_train, y_train, X_val, y_val, model_save_path="svm_model.pkl"):
    print("开始训练 SVM 模型...")
    model.train(X_train, y_train)

    # 计算训练和验证集上的准确率
    train_accuracy = model.model.score(X_train, y_train)
    val_accuracy = model.model.score(X_val, y_val)
    print(f"训练集准确率: {train_accuracy:.2f}, 验证集准确率: {val_accuracy:.2f}")

    # 保存模型
    model.save_model(model_save_path)
    print(f"SVM 模型已保存到 {model_save_path}")


# Step 4: Predict

# def predict_from_model(model, X_test):
#     print("开始预测...")
#     y_pred = model.predict(X_test)
#     return y_pred, None
def get_decision_scores(model, X):
    """
    计算 LinearSVC 的决策分数。

    :param model: 训练好的 LinearSVC 模型
    :param X: ndarray, 输入特征
    :return: ndarray, 决策分数
    """
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return np.dot(X, model.coef_.T) + model.intercept_
    else:
        raise AttributeError("模型缺少 coef_ 或 intercept_ 属性，无法计算决策分数。")


def predict_from_model(model, X_test, epsilon=0.1):
    """
    使用线性 SVM 模型进行预测，支持边界范围输出 0.5。

    :param model: 训练好的 SVM 模型
    :param X_test: ndarray, 测试集特征
    :param epsilon: float, 判别边界的宽度
    :return: ndarray, 预测结果（0, 1 或 0.5）
    """
    print("开始预测...")
    # 获取 SVM 的决策分数
    decision_scores = get_decision_scores(model, X_test)

    # 将分数映射到概率，并定义预测结果
    probabilities = 1 / (1 + np.exp(-decision_scores))
    y_pred = probabilities.copy()
    y_pred[probabilities > 0.5 + epsilon] = 1  # 高于边界范围归为 1
    y_pred[probabilities < 0.5 - epsilon] = 0  # 低于边界范围归为 0
    y_pred[np.abs(probabilities - 0.5) <= epsilon] = 0.5  # 边界范围内归为 0.5（不确定）
    y_pred = np.squeeze(y_pred)
    # print(probabilities)

    return y_pred, probabilities

# Step 5:  Evaluate


def predict_and_evaluate(model, X_test, y_test, epsilon=0.1):
    """
    使用模型预测并评估性能，同时去掉 y_pred=0.5 的位置和对应的 y_test。

    :param model: 训练好的 SVM 模型
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    :param epsilon: float, 判别边界的宽度
    :return: None
    """
    # 使用模型预测
    y_pred, probabilities = predict_from_model(model, X_test, epsilon=epsilon)

    # 去掉 y_pred=0.5 的位置和对应的 y_test
    valid_indices = y_pred != 0.5
    y_pred_filtered = y_pred[valid_indices]
    y_test_filtered = y_test[valid_indices]

    # 计算准确率
    accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
    print(f"测试集准确率: {accuracy:.2f}")

    # 打印去除的样本数
    removed_samples = len(y_test) - len(y_test_filtered)
    print(f"移除了 {removed_samples} 个样本，因为 y_pred=0.5。")

    # 混淆矩阵
    cm = confusion_matrix(y_test_filtered, y_pred_filtered)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()


# Step 5: Main function


def main():
    # File paths
    data_file_path = "LSTMdata.pkl"  # Path to your saved DataFrame
    model_save_path = "svm_model.pkl"  # Path where model will be saved

    os.makedirs("plots", exist_ok=True)

    # Load and prepare the data
    X, y = load_and_prepare_data(data_file_path)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 创建 SVM 模型
    svm_model = SVMmodels()

    # # 训练 SVM 模型
    # train_svm_model(svm_model, X_train, y_train, X_val, y_val, model_save_path)

    # 加载模型并预测
    svm_model.load_model(model_save_path)
    predict_and_evaluate(svm_model.model, X_test, y_test, epsilon=0.02)


if __name__ == "__main__":
    main()
