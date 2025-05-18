# src/dbscan_library.py
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class DBSCANLibrary:
    """Lớp triển khai DBSCAN sử dụng scikit-learn."""
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Khởi tạo DBSCAN.
        
        Args:
            eps (float): Bán kính láng giềng.
            min_samples (int): Số điểm tối thiểu trong láng giềng.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = None
    
    def fit(self, X):
        """
        Chạy thuật toán DBSCAN.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào [n_samples, n_features].
        
        Returns:
            np.ndarray: Nhãn cụm (-1 cho nhiễu).
        """
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
    
    def plot_k_distance(self, X, k=5, save_path="results/k_distance_plot.png"):
        """
        Vẽ k-distance plot để chọn eps.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            k (int): Số hàng xóm gần nhất.
            save_path (str): Đường dẫn lưu biểu đồ.
        """
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, k-1], axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(distances)
        plt.title("k-Distance Plot for DBSCAN")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{k}-th Nearest Neighbor Distance")
        plt.savefig(save_path)
        plt.close()
        print(f"k-Distance Plot đã lưu tại: {save_path}")

if __name__ == "__main__":
    data = pd.read_csv("data/processed_mall_customers.csv")
    X = data.values
    
    dbscan = DBSCANLibrary(eps=0.5, min_samples=5)
    labels = dbscan.fit(X)
    dbscan.plot_k_distance(X, k=5)
    
    np.save("results/dbscan_labels.npy", labels)
    print("Kết quả DBSCAN đã lưu tại results/dbscan_labels.npy")