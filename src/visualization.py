# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

class ClusterVisualizer:
    """Lớp trực quan hóa kết quả phân cụm."""
    
    def __init__(self):
        """Khởi tạo visualizer."""
        pass
    
    def plot_clusters(self, X, labels, centroids=None, method_name="", save_path="", cmap_name="tab10"):
        """
        Vẽ scatter plot cho kết quả phân cụm.
        
        Args:
            X (np.ndarray): Dữ liệu 2D.
            labels (np.ndarray): Nhãn cụm.
            centroids (np.ndarray, optional): Tọa độ tâm cụm.
            method_name (str): Tên phương pháp.
            save_path (str): Đường dẫn lưu biểu đồ.
            cmap_name (str): Tên của palette màu.
        """
        k = len(np.unique(labels))
        cmap = plt.get_cmap(cmap_name, k)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=50, alpha=0.8, edgecolor='k', linewidth=0.5)
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=250, linewidths=3, edgecolor='black', label='Centroids')
            for i, (x, y) in enumerate(centroids):
                plt.text(x, y, f'C{i}', fontsize=14, fontweight='bold', color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.title(f"Clustering Results - {method_name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, linestyle='--', alpha=0.3)
        handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(k)]
        handles.append(mpatches.Patch(color='red', label='Centroids'))
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(save_path)
        plt.close()
        print(f"Biểu đồ đã lưu tại: {save_path}")