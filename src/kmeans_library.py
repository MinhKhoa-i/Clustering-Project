from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def kmeans_library(X, k, random_state=42):
    """
    Chạy thuật toán K-Means sử dụng thư viện scikit-learn.
    
    Args:
        X (np.ndarray): Dữ liệu đầu vào, dạng ma trận [n_samples, n_features].
        k (int): Số lượng cụm.
        random_state (int): Seed để tái lập kết quả.
    
    Returns:
        tuple: (labels, centroids)
            - labels (np.ndarray): Nhãn cụm cho từng mẫu [n_samples].
            - centroids (np.ndarray): Tọa độ tâm cụm [k, n_features].
    """
    # Khởi tạo và chạy K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    return labels, centroids

def compute_inertia_library(X, labels, centroids):
    """
    Tính inertia (tổng bình phương khoảng cách từ điểm đến tâm cụm gần nhất).
    
    Args:
        X (np.ndarray): Dữ liệu đầu vào.
        labels (np.ndarray): Nhãn cụm.
        centroids (np.ndarray): Tâm cụm.
    
    Returns:
        float: Giá trị inertia.
    """
    inertia = 0
    for i in range(len(centroids)):
        points = X[labels == i]
        if len(points) > 0:
            distances = np.sqrt(((points - centroids[i]) ** 2).sum(axis=1))
            inertia += np.sum(distances ** 2)
    return inertia

if __name__ == "__main__":
    # Ví dụ chạy K-Means thư viện với dữ liệu đã tiền xử lý
    data = pd.read_csv("data/processed_mall_customers.csv")
    X = data.values  # Chuyển thành numpy array
    
    # Chạy K-Means
    k = 5  # Số cụm (có thể thay đổi)
    labels, centroids = kmeans_library(X, k=k, random_state=42)
    
    # Tính inertia
    inertia = compute_inertia_library(X, labels, centroids)
    print(f"Inertia: {inertia:.4f}")
    
    # Lưu kết quả
    np.save("results/kmeans_library_labels.npy", labels)
    np.save("results/kmeans_library_centroids.npy", centroids)
    print("Kết quả đã lưu tại results/kmeans_library_*.npy")