import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标。
    :param y_true: 真实标签列表 (0: 正常, 1: 不良坐姿)
    :param y_pred: 预测标签列表 (0: 正常, 1: 不良坐姿)
    :return: 包含各项指标的字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def print_report(metrics):
    """
    格式化打印评估结果。
    """
    print("-" * 30)
    print("坐姿监测系统评估报告")
    print("-" * 30)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.2%}")
    print(f"精确率 (Precision): {metrics['precision']:.2%}")
    print(f"召回率 (Recall):    {metrics['recall']:.2%}")
    print(f"F1 分数 (F1-Score): {metrics['f1']:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    # 示例数据 (用于测试脚本是否正常运行)
    # 假设我们有 10 张测试图片，真实情况和预测结果如下：
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5个正常，5个不良
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]  # 预测中：1个正常被误报为不良，1个不良未被发现
    
    results = calculate_metrics(y_true, y_pred)
    print_report(results)
