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
        Đánh giá phân cụm bằng nhiều độ đo.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào [n_samples, n_features].
            labels (np.ndarray): Nhãn cụm.
            centroids (np.ndarray, optional): Tọa độ tâm cụm (dùng cho K-Means).
            method_name (str): Tên phương pháp.
        
        Returns:
            dict: Kết quả đánh giá.
        """
        result = {}
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        
        # Chỉ tính các độ đo nếu có ít nhất 2 cụm (trừ nhiễu trong DBSCAN)
        if n_clusters > 1:
            # Silhouette Score
            silhouette = silhouette_score(X, labels)
            result["Silhouette Score"] = silhouette
            
            # Davies-Bouldin Index
            db_index = davies_bouldin_score(X, labels)
            result["Davies-Bouldin Index"] = db_index
            
            # Calinski-Harabasz Index
            ch_index = calinski_harabasz_score(X, labels)
            result["Calinski-Harabasz Index"] = ch_index
            
            print(f"{method_name} - Silhouette Score: {silhouette:.4f}, "
                  f"Davies-Bouldin Index: {db_index:.4f}, "
                  f"Calinski-Harabasz Index: {ch_index:.4f}")
            
            # MSE và RMSE (chỉ tính cho K-Means, khi có centroids)
            if centroids is not None:
                mse = self.compute_mse(X, labels, centroids)
                rmse = np.sqrt(mse)
                result["MSE"] = mse
                result["RMSE"] = rmse
                print(f"{method_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        else:
            print(f"{method_name} - Không đủ cụm để đánh giá (n_clusters={n_clusters})")
        
        self.results[method_name] = result
        return result
    
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