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

def calculate_performance(frame_times):
    """
    计算性能指标 (FPS 和 延迟)。
    :param frame_times: 每帧处理耗时的列表 (单位: 秒)
    :return: 包含性能指标的字典
    """
    avg_time = np.mean(frame_times)
    fps = 1 / avg_time if avg_time > 0 else 0
    return {
        "avg_latency_ms": avg_time * 1000,
        "avg_fps": fps
    }

def print_report(metrics, performance=None):
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
    
    if performance:
        print("-" * 30)
        print(f"平均延迟 (Latency): {performance['avg_latency_ms']:.2f} ms")
        print(f"平均刷新率 (FPS):   {performance['avg_fps']:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    # 示例数据
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    
    # 模拟 10 帧的处理耗时 (例如每帧 30ms-50ms)
    mock_frame_times = [0.032, 0.045, 0.038, 0.041, 0.035, 0.042, 0.039, 0.040, 0.037, 0.043]
    
    results = calculate_metrics(y_true, y_pred)
    perf = calculate_performance(mock_frame_times)
    print_report(results, perf)
