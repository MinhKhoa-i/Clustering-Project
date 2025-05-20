# src/evaluation.py
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

class ClusterEvaluator:
    """Lớp đánh giá chất lượng phân cụm."""
    
    def __init__(self):
        """Khởi tạo evaluator."""
        self.results = {}
    
    def evaluate(self, X, labels, centroids=None, method_name=""):
        """
        Đánh giá kết quả phân cụm.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            labels (np.ndarray): Nhãn cụm.
            centroids (np.ndarray, optional): Tâm cụm (cho K-Means).
            method_name (str): Tên phương pháp phân cụm.
        
        Returns:
            dict: Các chỉ số đánh giá.
        """
        metrics = {}
        
        # Silhouette Score
        metrics["Silhouette Score"] = silhouette_score(X, labels)
        
        # Davies-Bouldin Index
        metrics["Davies-Bouldin Index"] = davies_bouldin_score(X, labels)
        
        # Calinski-Harabasz Index
        metrics["Calinski-Harabasz Index"] = calinski_harabasz_score(X, labels)
        
        # MSE và RMSE (chỉ cho K-Means)
        if centroids is not None:
            mse = self.compute_mse(X, labels, centroids)
            metrics["MSE"] = mse
            metrics["RMSE"] = np.sqrt(mse)
        
        self.results[method_name] = metrics
        return metrics
    
    def compute_mse(self, X, labels, centroids):
        """
        Tính Mean Squared Error (MSE) dựa trên khoảng cách từ các điểm đến tâm cụm.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào [n_samples, n_features].
            labels (np.ndarray): Nhãn cụm.
            centroids (np.ndarray): Tọa độ tâm cụm [n_clusters, n_features].
        
        Returns:
            float: Giá trị MSE.
        """
        mse = 0
        n_samples = X.shape[0]
        for i in range(n_samples):
            cluster_idx = labels[i]
            if cluster_idx != -1:  # Bỏ qua điểm nhiễu (nếu có)
                distance = np.sum((X[i] - centroids[cluster_idx]) ** 2)
                mse += distance
        mse /= n_samples
        return mse
    
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