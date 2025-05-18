# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

class ClusterVisualizer:
    """Lớp trực quan hóa kết quả phân cụm."""
    
    def __init__(self):
        """Khởi tạo visualizer."""
        pass
    
    def plot_clusters(self, X, labels, centroids=None, method_name="", save_path=""):
        """
        Vẽ scatter plot cho kết quả phân cụm.
        
        Args:
            X (np.ndarray): Dữ liệu 2D.
            labels (np.ndarray): Nhãn cụm.
            centroids (np.ndarray, optional): Tọa độ tâm cụm.
            method_name (str): Tên phương pháp.
            save_path (str): Đường dẫn lưu biểu đồ.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200, 
                       linewidths=3, label="Centroids")
        plt.title(f"Clustering Results - {method_name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Biểu đồ đã lưu tại: {save_path}")