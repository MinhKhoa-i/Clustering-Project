# src/comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from .evaluation import ClusterEvaluator
from .kmeans_custom import KMeansCustom
from .kmeans_library import KMeansLibrary
from .dbscan_library import DBSCANLibrary

class ModelComparison:
    """Lớp so sánh và trực quan hóa các mô hình phân cụm."""
    
    def __init__(self):
        """Khởi tạo bộ so sánh mô hình."""
        self.evaluator = ClusterEvaluator()
    
    def run_all_models(self, X, k, max_iter=300, random_state=42, eps=0.5, min_samples=5):
        """
        Chạy tất cả các mô hình phân cụm.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            k (int): Số cụm cho K-Means.
            max_iter (int): Số vòng lặp tối đa cho K-Means.
            random_state (int): Seed cho K-Means.
            eps (float): Bán kính cho DBSCAN.
            min_samples (int): Số điểm tối thiểu cho DBSCAN.
        
        Returns:
            tuple: (labels_dict, centroids_dict, metrics_dict)
        """
        labels_dict = {}
        centroids_dict = {}
        
        # KMeans Custom
        kmeans_custom = KMeansCustom(k=k, max_iters=max_iter, random_state=random_state)
        labels_custom, centroids_custom = kmeans_custom.fit(X)
        labels_dict["KMeans (Viết tay)"] = labels_custom
        centroids_dict["KMeans (Viết tay)"] = centroids_custom
        
        # KMeans Library
        kmeans_lib = KMeansLibrary(k=k, random_state=random_state)
        labels_lib, centroids_lib = kmeans_lib.fit(X)
        labels_dict["KMeans (Thư viện)"] = labels_lib
        centroids_dict["KMeans (Thư viện)"] = centroids_lib
        
        # DBSCAN
        dbscan = DBSCANLibrary(eps=eps, min_samples=min_samples)
        labels_dbscan = dbscan.fit(X)
        labels_dict["DBSCAN"] = labels_dbscan
        centroids_dict["DBSCAN"] = None
        
        # Tính toán các chỉ số đánh giá
        metrics_dict = {}
        for name, labels in labels_dict.items():
            centroids = centroids_dict[name]
            metrics = self.evaluator.evaluate(X, labels, centroids, method_name=name)
            metrics_dict[name] = metrics
        
        return labels_dict, centroids_dict, metrics_dict
    
    def plot_comparison(self, X, labels_dict, centroids_dict=None):
        """
        Vẽ biểu đồ so sánh kết quả của các mô hình.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            labels_dict (dict): Dictionary chứa nhãn của từng mô hình.
            centroids_dict (dict, optional): Dictionary chứa centroids của từng mô hình.
        
        Returns:
            matplotlib.figure.Figure: Biểu đồ so sánh.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Nếu dữ liệu nhiều hơn 2 chiều, sử dụng PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)
            if centroids_dict:
                centroids_dict = {k: pca.transform(v) if v is not None else None 
                                for k, v in centroids_dict.items()}
        else:
            X2 = X
        
        # Vẽ từng mô hình
        for i, (name, labels) in enumerate(labels_dict.items()):
            ax = axes[i]
            cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))
            
            # Vẽ các điểm dữ liệu
            scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=cmap, 
                               s=50, alpha=0.8, edgecolor='k', linewidth=0.5)
            
            # Vẽ centroids nếu có
            if centroids_dict and centroids_dict[name] is not None:
                centroids = centroids_dict[name]
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', 
                          marker='X', s=200, linewidths=3, edgecolor='black')
                for j, (x, y) in enumerate(centroids):
                    ax.text(x, y, f'C{j}', fontsize=12, fontweight='bold',
                           color='black', ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            ax.set_title(f"{name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Thêm legend
            handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') 
                      for i in range(len(np.unique(labels)))]
            if centroids_dict and centroids_dict[name] is not None:
                handles.append(mpatches.Patch(color='red', label='Centroids'))
            ax.legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def get_comparison_explanation(self):
        """
        Trả về giải thích về kết quả so sánh.
        
        Returns:
            str: Giải thích kết quả so sánh.
        """
        return """
        **Giải thích kết quả so sánh:**
        
        1. **KMeans (Viết tay) vs KMeans (Thư viện):**
           - Cả hai đều sử dụng cùng thuật toán K-Means
           - KMeans thư viện thường tối ưu hơn về mặt tính toán
           - Kết quả có thể khác nhau do khởi tạo ngẫu nhiên
        
        2. **KMeans vs DBSCAN:**
           - KMeans yêu cầu số cụm cố định, DBSCAN tự động xác định
           - DBSCAN tốt hơn với dữ liệu có hình dạng phức tạp
           - KMeans tốt hơn với dữ liệu có dạng hình cầu
        
        3. **Chỉ số đánh giá:**
           - Silhouette Score: càng cao càng tốt
           - Davies-Bouldin Index: càng thấp càng tốt
           - Calinski-Harabasz Index: càng cao càng tốt
        """ 