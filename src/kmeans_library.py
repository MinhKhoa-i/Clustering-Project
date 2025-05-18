# src/kmeans_library.py
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeansLibrary:
    """Lớp triển khai K-Means sử dụng thư viện scikit-learn."""
    
    def __init__(self, k, random_state=42):
        """
        Khởi tạo K-Means.
        
        Args:
            k (int): Số lượng cụm.
            random_state (int): Seed để tái lập kết quả.
        """
        self.k = k
        self.random_state = random_state
        self.model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        self.labels_ = None
        self.centroids_ = None
    
    def fit(self, X):
        """
        Chạy thuật toán K-Means.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào [n_samples, n_features].
        
        Returns:
            tuple: (labels, centroids)
        """
        self.labels_ = self.model.fit_predict(X)
        self.centroids_ = self.model.cluster_centers_
        return self.labels_, self.centroids_
    
    def compute_inertia(self, X):
        """
        Tính inertia (tổng bình phương khoảng cách từ điểm đến tâm cụm gần nhất).
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
        
        Returns:
            float: Giá trị inertia.
        """
        inertia = 0
        for i in range(self.k):
            points = X[self.labels_ == i]
            if len(points) > 0:
                distances = np.sqrt(((points - self.centroids_[i]) ** 2).sum(axis=1))
                inertia += np.sum(distances ** 2)
        return inertia
    
    def plot_elbow_method(self, X, max_k=10, save_path="results/elbow_plot_library.png"):
        """
        Vẽ biểu đồ Elbow để chọn số cụm tối ưu.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            max_k (int): Số cụm tối đa để kiểm tra.
            save_path (str): Đường dẫn lưu biểu đồ.
        """
        inertias = []
        for k in range(1, max_k + 1):
            model = KMeansLibrary(k, self.random_state)
            model.fit(X)
            inertia = model.compute_inertia(X)
            inertias.append(inertia)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title("Elbow Method for Optimal k (Library)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.savefig(save_path)
        plt.close()
        print(f"Biểu đồ Elbow đã lưu tại: {save_path}")

if __name__ == "__main__":
    data = pd.read_csv("data/processed_mall_customers.csv")
    X = data.values
    
    kmeans = KMeansLibrary(k=5, random_state=42)
    labels, centroids = kmeans.fit(X)
    inertia = kmeans.compute_inertia(X)
    print(f"Inertia: {inertia:.4f}")
    
    kmeans.plot_elbow_method(X, max_k=10)
    
    np.save("results/kmeans_library_labels.npy", labels)
    np.save("results/kmeans_library_centroids.npy", centroids)
    print("Kết quả đã lưu tại results/kmeans_library_*.npy")