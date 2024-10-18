import numpy as np
import torch
import pandas as pd  # 引入 pandas
from vibration_analysis import VibrationAnalysis
from conv_lstm_model import ConvLSTMModel
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    precision_recall_curve,
    accuracy_score,     # 增加這一行
    precision_score,    # 增加這一行
    recall_score,       # 增加這一行
    f1_score            # 增加這一行
)
import matplotlib.pyplot as plt
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 健康狀態分類與評估指標計算


def health_status_classifier(predictions):
    """
    將模型預測值轉換為健康狀態類別。
    參數:
    - predictions: 模型輸出 (數值)
    返回:
    - 健康狀態 ('正常' 或 '異常')
    """
    return '異常' if predictions > 0.5 else '正常'


def calculate_metrics(true_labels, predictions):
    """
    計算模型預測的各種評估指標。
    參數:
    - true_labels: 真實標籤
    - predictions: 模型預測值
    返回:
    - 各種評估指標 (字典)
    """
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# 繪製混淆矩陣


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.title('混淆矩陣')
    plt.show()

# 繪製ROC曲線


def plot_roc_curve(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC曲線 (面積 = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假陽性率')
    plt.ylabel('真陽性率')
    plt.title('接收者操作特徵曲線 (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# 繪製精確度-召回率曲線


def plot_precision_recall_curve(true_labels, scores):
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    plt.figure()
    plt.step(recall, precision, where='post', color='blue', label='精確度-召回率曲線')
    plt.xlabel('召回率')
    plt.ylabel('精確度')
    plt.title('精確度-召回率曲線')
    plt.show()

# 繪製分類報告


def plot_classification_report(true_labels, predictions):
    report = classification_report(true_labels, predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, fmt=".2f", cmap='Blues')
    plt.title('分類報告')
    plt.xlabel('指標')
    plt.ylabel('類別')
    plt.show()

# 繪製預測 vs 實際散點圖


def plot_predictions_vs_actual(predictions, true_labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(true_labels, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 對角線
    plt.xlabel('真實標籤')
    plt.ylabel('預測標籤')
    plt.title('預測 vs 實際')
    plt.show()

# 繪製預測誤差分佈直方圖


def plot_prediction_errors(predictions, true_labels):
    errors = predictions - true_labels
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, color='skyblue', alpha=0.7)
    plt.xlabel('預測誤差')
    plt.ylabel('頻數')
    plt.title('預測誤差分佈')
    plt.show()

# 繪製各狀態的分佈比較柱狀圖


def plot_distribution_comparison(labels, title='狀態分佈比較'):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels)
    plt.xlabel('狀態')
    plt.ylabel('數量')
    plt.title(title)
    plt.show()


# 初始化參數
input_size = 10
hidden_size = 50
num_layers = 3  # 增加層數
output_size = 1
batch_size = 32
sequence_length = 10

# 生成振動數據
vibration_analysis = VibrationAnalysis(
    input_size, hidden_size, num_layers, output_size)
vibration_data = vibration_analysis.generate_complex_vibration_data(
    harmonics=5, noise_level=0.1, wear_factor=0.1)
features_time = vibration_analysis.calculate_time_domain_features(
    vibration_data)
features_freq = vibration_analysis.calculate_frequency_domain_features(
    vibration_data)

# 模型初始化
model = ConvLSTMModel(input_size, hidden_size, num_layers, output_size)
model.eval()

# 假設性預測 (隨機生成輸入)
with torch.no_grad():
    input_tensor = torch.tensor(np.random.rand(
        batch_size, sequence_length, input_size), dtype=torch.float32)
    output = model(input_tensor)

# 健康狀態分類
status = health_status_classifier(output.mean().item())
print(f"預測健康狀態: {status}")

# 假設性真實標籤與預測標籤 (隨機生成)
true_labels = np.random.randint(0, 2, batch_size)
predictions = (output.detach().numpy() > 0.5).astype(int).flatten()  # 將預測值二值化

# 計算評估指標
metrics = calculate_metrics(true_labels, predictions)
print(f"計算的評估指標: {metrics}")

# 繪製進階圖表
plot_confusion_matrix(true_labels, predictions)
plot_roc_curve(true_labels, output.detach().numpy())
plot_precision_recall_curve(true_labels, output.detach().numpy())
plot_classification_report(true_labels, predictions)
plot_predictions_vs_actual(predictions, true_labels)
plot_prediction_errors(predictions, true_labels)
plot_distribution_comparison(predictions)

# 模擬健康指數的退化
health_index = vibration_analysis.simulate_health_index_degradation(
    initial_health=1.0, degradation_rate=0.05, time_steps=100)
print(f"模擬的健康指數: {health_index}")
