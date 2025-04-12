# src/dbscan_library.py
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

def dbscan_library(X, eps=0.5, min_samples=5):
    """
    Chạy thuật toán DBSCAN sử dụng thư viện scikit-learn.
    
    Args:
        X (np.ndarray): Dữ liệu đầu vào, dạng ma trận [n_samples, n_features].
        eps (float): Bán kính láng giềng.
        min_samples (int): Số điểm tối thiểu trong láng giềng để tạo cụm.
    
    Returns:
        np.ndarray: Nhãn cụm (-1 cho nhiễu, 0, 1, ... cho các cụm).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

if __name__ == "__main__":
    # Ví dụ chạy DBSCAN
    data = pd.read_csv("data/processed_mall_customers.csv")
    X = data.values
    
    # Chạy DBSCAN
    eps = 0.5
    min_samples = 5
    labels = dbscan_library(X, eps=eps, min_samples=min_samples)
    
    # Lưu kết quả
    np.save("results/dbscan_labels.npy", labels)
    print("Kết quả DBSCAN đã lưu tại results/dbscan_labels.npy")