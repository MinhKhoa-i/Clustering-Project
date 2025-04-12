# src/evaluation.py
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

def evaluate_clustering(X, labels, method_name):
    """
    Đánh giá chất lượng phân cụm bằng Silhouette Score và Davies-Bouldin Index.
    
    Args:
        X (np.ndarray): Dữ liệu đầu vào.
        labels (np.ndarray): Nhãn cụm.
        method_name (str): Tên phương pháp (ví dụ: K-Means Custom).
    
    Returns:
        dict: Kết quả đánh giá {metric_name: value}.
    """
    results = {}
    
    # Chỉ tính nếu có nhiều hơn 1 cụm và không có nhiễu (-1)
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        results["Silhouette Score"] = silhouette
        results["Davies-Bouldin Index"] = db_index
        print(f"{method_name} - Silhouette Score: {silhouette:.4f}, Davies-Bouldin Index: {db_index:.4f}")
    else:
        print(f"{method_name} - Không đủ cụm để đánh giá (n_clusters={n_clusters})")
    
    return results

def save_evaluation_results(results_dict, output_path="results/evaluation_metrics.txt"):
    """
    Lưu kết quả đánh giá vào file.
    
    Args:
        results_dict (dict): Kết quả đánh giá từ các phương pháp.
        output_path (str): Đường dẫn file đầu ra.
    """
    with open(output_path, "w") as f:
        for method, metrics in results_dict.items():
            f.write(f"{method}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
    print(f"Kết quả đánh giá đã lưu tại: {output_path}")