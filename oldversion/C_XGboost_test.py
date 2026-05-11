from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import pandas_ta as ta
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存和加载缩放器
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


class XGBoostModel:
    def __init__(self, params=None, num_boost_round=100):
        """ 
        初始化XGBoost模型类。
         :param params: dict, XGBoost的参数字典。例如：{'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.3}
        :param num_boost_round: int, boosting的轮数
        """
        self.params = params if params else {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.3}
        self.num_boost_round = num_boost_round
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        训练XGBoost模型。

        :param X_train: np.array or pd.DataFrame, 训练特征
        :param y_train: np.array or pd.Series, 训练标签
        :param X_val: np.array or pd.DataFrame, 验证特征（可选）
        :param y_val: np.array or pd.Series, 验证标签（可选）
        :param early_stopping_rounds: int, 早停的轮数（仅在提供验证集时有效）
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
            self.model = xgb.train(
                self.params, dtrain, num_boost_round=self.num_boost_round, evals=evals,
                early_stopping_rounds=early_stopping_rounds, verbose_eval=True
            )
        else:
            self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        :param X: np.array or pd.DataFrame, 输入特征
        :return: np.array, 预测结果
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def evaluate(self, X, y, metric='accuracy'):
        """
        评估模型性能。

        :param X: np.array or pd.DataFrame, 输入特征
        :param y: np.array or pd.Series, 真实标签
        :param metric: str, 性能指标（'accuracy' 或 'rmse'）
        :return: float, 性能分数
        """
        y_pred = self.predict(X)
        if metric == 'accuracy':
            y_pred_binary = np.round(y_pred)  # 适用于分类问题
            return accuracy_score(y, y_pred_binary)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y, y_pred))
        else:
            raise ValueError("Unsupported metric. Use 'accuracy' or 'rmse'.")

    def save_model(self, path):
        """
        保存模型到文件。

        :param path: str, 模型保存路径
        """
        self.model.save_model(path)

    def load_model(self, path):
        """
        从文件加载模型。

        :param path: str, 模型文件路径
        """
        self.model = xgb.Booster()
        self.model.load_model(path)


def load_and_prepare_data(file_path, scaler_path=None, save_scaler=True, normalization='global', window_size=1000):
    """
    加载数据并处理特征，同时应用预定义缩放规则。
    :param file_path: str, 数据文件路径
    :param scaler_path: str, 保存或加载缩放器的路径。如果为 None，则不保存。
    :param save_scaler: bool, 是否保存新的缩放器（仅在未提供 scaler_path 时生效）。
    :param normalization: str, 归一化方法（'global', 'sliding'）。
    :param window_size: int, 滑动窗口大小，仅在 'sliding' 方法中使用。
    :return: np.array, 特征矩阵 X 和标签 y
    """
    # 加载数据
    df = pd.read_pickle(file_path)  # 假设数据为 DataFrame 格式
    X, y = load_and_prepare_data_real(df)  # 提取特征和标签

    # 应用归一化方法
    if normalization == 'sliding':
        print(f'Use Sliding win:{window_size}')
        X_scaled = sliding_window_normalize(X, window_size=window_size)
        y = y[window_size - 1:]  # 调整标签大小，保持与滑动窗口对齐
    elif normalization == 'global':
        if scaler_path and os.path.exists(scaler_path):
            # 加载已有缩放器
            scaler = joblib.load(scaler_path)
            print(f"缩放器已加载: {scaler_path}")
        else:
            # 创建新的全局缩放器
            scaler = MinMaxScaler()
            scaler.fit(X)
            if save_scaler and scaler_path:
                joblib.dump(scaler, scaler_path)
                print(f"缩放器已保存: {scaler_path}")
        X_scaled = scaler.transform(X)
    else:
        raise ValueError("Unsupported normalization method. Use 'sliding' or 'global'.")

    return X_scaled, y


def sliding_window_normalize(X, window_size=1000):
    """
    使用滑动窗口对时间序列数据进行归一化，并只提取每个窗口的最后一个时间步特征。
    :param X: list or np.array, 输入特征矩阵
    :param window_size: int, 滑动窗口大小
    :return: np.array, 归一化后的特征矩阵，仅保留窗口内最后一个时间步
    """
    X = np.array(X)  # 确保输入为 NumPy 数组
    normalized_features = []

    for i in range(window_size - 1, len(X)):
        if i % 100 == 0:
            print(f'{i}|{len(X)}')
        # 取窗口数据
        window = X[i - window_size + 1:i + 1]

        # 计算最小值和最大值
        min_val = np.min(window, axis=0)
        max_val = np.max(window, axis=0)

        # 归一化窗口数据
        normalized_window = (window - min_val) / (max_val - min_val + 1e-8)

        # 提取最后一个时间步
        normalized_features.append(normalized_window[-1])

    return np.array(normalized_features)


def load_and_prepare_data_realx(df):
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

    # Parabolic SAR# Parabolic SAR
    psar = ta.psar(high=df['high'], low=df['low'], close=df['close'])
    # 将相关的列添加到主 DataFrame
    df['PSAR_long'] = psar['PSARl_0.02_0.2']  # Long方向的PSAR
    df['PSAR_short'] = psar['PSARs_0.02_0.2']  # Short方向的PSAR
    df['PSAR_af'] = psar['PSARaf_0.02_0.2']  # 加速因子
    df['PSAR_reversal'] = psar['PSARr_0.02_0.2']  # 反转点

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
    # 使用优化函数
    df['rolling_skew'] = df.ta.skew(close=df['log_return'], length=20)
    df['rolling_kurtosis'] = df.ta.kurtosis(close=df['log_return'], length=20)

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
        'PSAR_af', 'PSAR_reversal',
        'ADX', 'PDI', 'NDI', 'DX',
        'Volume_MA5', 'Volume_MA20', 'price_to_volume', 'volume_acceleration',
        'OBV', 'OBV_change',
        'rolling_skew', 'rolling_kurtosis',
        'is_hammer'
    ]
    for lag in range(1, 6):
        features.append(f'MA20_lag_{lag}')

    df['position_signal'] = df['position_signal'].map({'long': 1, 'short': 0})
    # df.dropna(subset=features + ['position_signal'], inplace=True)
    X = df[features].values
    # y = df['position_signal'].values
    return X


def load_and_prepare_data_realy(df):
    df['position_signal'] = df['position_signal'].map({'long': 1, 'short': 0})
    y = df['position_signal'].values
    return y


def load_and_prepare_data_real(df):
    X = load_and_prepare_data_realx(df)
    y = load_and_prepare_data_realy(df)
    return X, y


def custom_threshold(y_pred, low_threshold=0.3, high_threshold=0.7):
    """
    根据自定义的阈值范围对预测结果进行分类。

    :param y_pred: np.array, XGBoost 的预测概率
    :param low_threshold: float, 下限阈值（低于此值为 0）
    :param high_threshold: float, 上限阈值（高于此值为 1）
    :return: np.array, 应用阈值后的分类结果
    """
    y_custom = np.where(y_pred < low_threshold, 0,  # 低于 low_threshold 分为 0
                        np.where(y_pred > high_threshold, 1, 0.5))  # 高于 high_threshold 分为 1，其余为 0.5
    return y_custom


def get_position(df):
    xgb_model = XGBoostModel()
    model_save_path = "xgboost_model.json"
    X_test = load_and_prepare_data_realx(df)

    xgb_model.load_model(model_save_path)
    y_pred_binary = xgb_model.predict(X_test)
    y_pred = custom_threshold(y_pred_binary, low_threshold=0.3, high_threshold=0.7)
    return y_pred


if __name__ == "__main__":
    # 文件路径
    test_data_file_path = "LSTMTestData.pkl"
    model_save_path = "xgboost_model.json"  # Path where model will be saved
    scaler_save_path = "scaler.pkl"

    df = pd.read_pickle("market_data_DOGE_new_with_position.pkl")  # 假设数据为 DataFrame 格式
    y = get_position(df)  # 提取特征和标签
    print(y)
