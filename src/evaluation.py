# src/evaluation.py
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

class ClusterEvaluator:
    """Lớp đánh giá chất lượng phân cụm."""
    
    def __init__(self):
        """Khởi tạo evaluator."""
        self.results = {}
    
    def evaluate(self, X, labels, method_name):
        """
        Đánh giá phân cụm bằng Silhouette Score và Davies-Bouldin Index.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            labels (np.ndarray): Nhãn cụm.
            method_name (str): Tên phương pháp.
        
        Returns:
            dict: Kết quả đánh giá.
        """
        result = {}
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            db_index = davies_bouldin_score(X, labels)
            result["Silhouette Score"] = silhouette
            result["Davies-Bouldin Index"] = db_index
            print(f"{method_name} - Silhouette Score: {silhouette:.4f}, "
                  f"Davies-Bouldin Index: {db_index:.4f}")
        else:
            print(f"{method_name} - Không đủ cụm để đánh giá (n_clusters={n_clusters})")
        
        self.results[method_name] = result
        return result
    
    def save_results(self, output_path="results/evaluation_metrics.txt"):
        """
        Lưu kết quả đánh giá.
        
        Args:
            output_path (str): Đường dẫn file đầu ra.
        """
        with open(output_path, "w") as f:
            for method, metrics in self.results.items():
                f.write(f"{method}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        print(f"Kết quả đánh giá đã lưu tại: {output_path}")