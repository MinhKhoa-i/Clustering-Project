# src/kmeans_custom.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class KMeansCustom:
    """Lớp triển khai K-Means tự viết."""
    
    def __init__(self, k, max_iters=100, random_state=None):
        """
        Khởi tạo K-Means tự viết.
        
        Args:
            k (int): Số lượng cụm.
            max_iters (int): Số vòng lặp tối đa.
            random_state (int): Seed để tái lập kết quả.
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
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
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids_ = X[idx]
        
        for _ in range(self.max_iters):
            distances = np.sqrt(((X[:, np.newaxis] - self.centroids_) ** 2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(self.centroids_)
            for i in range(self.k):
                if np.sum(self.labels_ == i) > 0:
                    new_centroids[i] = np.mean(X[self.labels_ == i], axis=0)
                else:
                    new_centroids[i] = self.centroids_[i]
            
            if np.all(self.centroids_ == new_centroids):
                break
            
            self.centroids_ = new_centroids
        
        return self.labels_, self.centroids_
    
    def compute_inertia(self, X):
        """
        Tính inertia.
        
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
    
    def plot_elbow_method(self, X, max_k=10, save_path="results/elbow_plot_custom.png"):
        """
        Vẽ biểu đồ Elbow.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            max_k (int): Số cụm tối đa.
            save_path (str): Đường dẫn lưu biểu đồ.
        """
        inertias = []
        for k in range(1, max_k + 1):
            model = KMeansCustom(k, self.max_iters, self.random_state)
            model.fit(X)
            inertia = model.compute_inertia(X)
            inertias.append(inertia)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title("Elbow Method for Optimal k (Custom)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.savefig(save_path)
        plt.close()
        print(f"Biểu đồ Elbow đã lưu tại: {save_path}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed_mall_customers.csv"
    data = pd.read_csv(file_path)
    X = data.values
    
    kmeans = KMeansCustom(k=5, random_state=42)
    labels, centroids = kmeans.fit(X)
    inertia = kmeans.compute_inertia(X)
    print(f"Inertia: {inertia:.4f}")
    
    kmeans.plot_elbow_method(X, max_k=10)
    
    np.save("results/kmeans_custom_labels.npy", labels)
    np.save("results/kmeans_custom_centroids.npy", centroids)
    print("Kết quả đã lưu tại results/kmeans_custom_*.npy")