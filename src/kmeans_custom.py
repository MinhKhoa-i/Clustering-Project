import numpy as np

def kmeans_custom(X, k, max_iters=100, random_state=None):
    """
    Triển khai thuật toán K-Means tự viết.
    
    Args:
        X (np.ndarray): Dữ liệu đầu vào, dạng ma trận [n_samples, n_features].
        k (int): Số lượng cụm.
        max_iters (int): Số vòng lặp tối đa.
        random_state (int): Seed để tái lập kết quả (nếu có).
    
    Returns:
        tuple: (labels, centroids)
            - labels (np.ndarray): Nhãn cụm cho từng mẫu [n_samples].
            - centroids (np.ndarray): Tọa độ tâm cụm [k, n_features].
    """
    # Thiết lập seed để tái lập kết quả
    if random_state is not None:
        np.random.seed(random_state)
    
    # Khởi tạo ngẫu nhiên các tâm cụm
    n_samples = X.shape[0]
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx]
    
    for _ in range(max_iters):
        # Bước 1: Gán nhãn dựa trên khoảng cách đến tâm cụm
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))  # [n_samples, k]
        labels = np.argmin(distances, axis=1)  # Nhãn cụm gần nhất
        
        # Bước 2: Cập nhật tâm cụm
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(labels == i) > 0:  # Chỉ cập nhật nếu cụm có điểm
                new_centroids[i] = np.mean(X[labels == i], axis=0)
            else:
                new_centroids[i] = centroids[i]  # Giữ nguyên nếu cụm rỗng
        
        # Bước 3: Kiểm tra hội tụ
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

def compute_inertia(X, labels, centroids):
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
    # Ví dụ chạy K-Means tự viết với dữ liệu đã tiền xử lý
    import pandas as pd
    
    # Tải dữ liệu đã tiền xử lý
    data = pd.read_csv("data/processed_mall_customers.csv")
    X = data.values  # Chuyển thành numpy array
    
    # Chạy K-Means
    k = 5  # Số cụm (có thể thay đổi)
    labels, centroids = kmeans_custom(X, k=k, random_state=42)
    
    # Tính inertia
    inertia = compute_inertia(X, labels, centroids)
    print(f"Inertia: {inertia:.4f}")
    
    # Lưu kết quả
    np.save("results/kmeans_custom_labels.npy", labels)
    np.save("results/kmeans_custom_centroids.npy", centroids)
    print("Kết quả đã lưu tại results/kmeans_custom_*.npy")