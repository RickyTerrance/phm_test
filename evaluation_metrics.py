import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 健康狀態分類函數


def health_status_classifier(predictions):
    """
    將模型預測值轉換為健康狀態類別。
    參數:
    - predictions: 模型輸出 (數值)
    返回:
    - 健康狀態 ('正常' 或 '異常')
    """
    if predictions > 0.5:
        return '異常'
    else:
        return '正常'

# 計算評估指標


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
