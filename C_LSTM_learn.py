import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Step 1: Data Loading and Preprocessin

def load_and_prepare_data(file_path):
    """
    Load the dataset and preprocess it for LSTM input.
    :param file_path: Path to the saved DataFrame file (e.g., CSV, Pickle)
    :return: Prepared training and test sets
    """
    df = pd.read_pickle(file_path)  # Or use pd.read_pickle(file_path) if in pickle format

    # 计算对数收益率
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_return'], inplace=True)  # 删除NaN值

    # 定义所需的特征列（包括扩展的特征）
    features = ['open', 'high', 'low', 'close', 'volume', 'log_return']
    # Select the features for LSTM, we'll use 'close' and 'position_signal' as targets
    # features = ['open', 'high', 'low', 'close', 'volume']
    target = ['position_signal']

    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])
    # window_size = 60
    # for feature in features:
    #     # 基于滑动窗口计算均值和标准差
    #     df[f'{feature}_norm'] = (df[feature] - df[feature].rolling(window_size).mean()) / \
    #                             (df[feature].rolling(window_size).std() + 1e-8)

    # 将 position_signal 映射为 long=1, short=0, 去掉 'none'
    df['position_signal'] = df['position_signal'].map({'long': 1, 'short': 0})

    # 删除 'none' 状态的数据行
    # df.dropna(subset=['position_signal'], inplace=True)

    df.dropna(inplace=True)

    # Prepare sequences for LSTM (X: features, y: position_signal)
    # normalized_features = [f'{feature}_norm' for feature in features]
    # X = df[normalized_features].values  # 特征矩阵
    X = df[features].values
    y = df[target].values.flatten()  # Ensure target is 1D

    return X, y


def create_sequences(X, y, time_steps=60):
    """
    Create sequences of data for LSTM input.
    :param X: Input features
    :param y: Target values (position_signal)
    :param time_steps: Number of time steps to look back in the LSTM
    :return: Sequences of X and y
    """
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X[i - time_steps:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


# Step 2: LSTM Model in PyTorch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)  # 增加一个全连接层
        self.fc2 = nn.Linear(50, 1)  # 输出为 1 个节点
        self.sigmoid = nn.Sigmoid()  # Sigmoid ensures the output is in [0,1]

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get output of last time step
        out = self.fc1(lstm_out)
        out = torch.relu(out)  # 使用 ReLU 激活函数
        out = self.fc2(out)
        return self.sigmoid(out)


# Step 3: Train the LSTM Model in PyTorch


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, model_save_path="lstm_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型转移到 GPU

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

    # 单卡训练，不需要设置多进程
    print('Start Loading.')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('Start Training.')
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # 使用 tqdm 包裹 train_loader 添加进度条
        # for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).flatten()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        # 使用 tqdm 包裹 val_loader 添加进度条
        with torch.no_grad():
            # for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).flatten()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots/training_validation_loss.png")
    plt.close()


# Step 4: Predict and Evaluate

def predict_and_evaluate(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    probabilities = []
    with torch.no_grad():
        for X_batch in torch.Tensor(X_test):
            X_batch = X_batch.to(device)
            outputs = model(X_batch.unsqueeze(0)).flatten()
            probabilities.append(outputs.item())  # Store probabilities for ROC curve
            predictions.append(1 if outputs.item() > 0.5 else 0)

    # 计算准确率
    accuracy = np.mean(np.array(predictions) == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # 绘制并保存 ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("plots/roc_curve.png")
    plt.close()

    return predictions, probabilities


# Step 5: Main function

def main():
    # File paths
    data_file_path = "LSTMdata.pkl"  # Path to your saved DataFrame
    model_save_path = "lstm_model.pth"  # Path where model will be saved

    os.makedirs("plots", exist_ok=True)

    # Load and prepare the data
    X, y = load_and_prepare_data(data_file_path)
    # print(X)

    # Create sequences for LSTM
    time_steps = 60  # Lookback period for LSTM
    X_seq, y_seq = create_sequences(X, y, time_steps)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define model parameters
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    # Create LSTM model
    model = LSTMModel(input_size, num_layers, hidden_size)

    # Train the LSTM model and save it
    train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=400, batch_size=64, model_save_path=model_save_path)

    # Predict and evaluate the model on test data
    predictions, probabilities = predict_and_evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
