import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


class ConvLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        """
        初始化卷積LSTM模型。

        參數:
        - input_size: 輸入特徵的大小（特徵數量）
        - hidden_size: 隱藏層的大小
        - num_layers: LSTM的層數
        - output_size: 輸出特徵的大小
        - dropout_rate: 隱藏層之間的 dropout 機率，默認為0.3
        """
        super(ConvLSTMModel, self).__init__()

        # 卷積層
        self.conv = nn.Conv1d(in_channels=input_size,
                              out_channels=hidden_size, kernel_size=3, padding=1)

        # 定義LSTM層
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(hidden_size if i == 0 else hidden_size,
                     hidden_size,
                     batch_first=True,
                     dropout=dropout_rate if i < num_layers - 1 else 0)  # 在最後一層不使用dropout
             for i in range(num_layers)]
        )

        # 全連接層
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)  # 在全連接層前添加 dropout
        self.relu = nn.ReLU()  # 添加 ReLU 激活函數

    def forward(self, x):
        """
        定義前向傳播流程。

        參數:
        - x: 輸入的數據，形狀為 (batch_size, sequence_length, input_size)

        返回:
        - predictions: 模型的預測值，形狀為 (batch_size, output_size)
        """
        # 使用卷積層提取特徵
        # 轉換形狀為 (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        x = self.conv(x)  # 應用卷積層
        x = x.transpose(1, 2)  # 轉換回 (batch_size, sequence_length, hidden_size)

        for lstm in self.lstm_layers:
            x, _ = lstm(x)  # 對於每一層LSTM，獲取輸出

        predictions = self.fc(x[:, -1, :])  # 取最後一個時間步的輸出
        predictions = self.dropout(predictions)  # 應用dropout
        predictions = self.relu(predictions)  # 應用激活函數
        return predictions  # 返回預測結果


# 用法示例
if __name__ == "__main__":
    # 假設參數
    input_size = 10
    hidden_size = 50
    num_layers = 3
    output_size = 1
    model = ConvLSTMModel(input_size, hidden_size, num_layers, output_size)

    # 隨機生成輸入數據
    input_tensor = torch.randn(32, 10, input_size)  # 假設批次大小為32，序列長度為10
    output = model(input_tensor)  # 獲取模型預測

    # 假設輸出為二進制分類，將預測結果轉換為0或1
    predicted_labels = (output.detach().numpy() > 0.5).astype(int)

    # 假設真實標籤
    true_labels = torch.randint(0, 2, (32, output_size)).numpy()

    # 計算準確率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Model output shape: {output.shape}")  # 輸出形狀應該是 (32, 1)
    print(f"Accuracy: {accuracy:.2f}")  # 輸出準確率
