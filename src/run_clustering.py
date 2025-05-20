# src/run_clustering.py
import pandas as pd
import numpy as np
from kmeans_library import KMeansLibrary
from kmeans_custom import KMeansCustom
from dbscan_library import DBSCANLibrary
from preprocessing import DataPreprocessor
from visualization import ClusterVisualizer
from evaluation import ClusterEvaluator

def analyze_clusters(data, labels, output_path="results/cluster_summary.csv"):
    """Phân tích đặc điểm trung bình của các cụm."""
    data["Cluster"] = labels
    cluster_summary = data.groupby("Cluster").mean()
    cluster_counts = data["Cluster"].value_counts().sort_index()
    print("Số điểm trong mỗi cụm:\n", cluster_counts)
    print("Cluster Summary:\n", cluster_summary)
    cluster_summary.to_csv(output_path)
    print(f"Tóm tắt cụm đã lưu tại: {output_path}")

def run_clustering():
    # Tiền xử lý dữ liệu
    preprocessor = DataPreprocessor(use_pca=True, n_components=2)
    preprocessor.preprocess(
        file_path="data/Mall_Customers.csv",
        output_path_scaled="data/processed_mall_customers.csv",
        output_path_pca="data/mall_customers_pca.csv"
    )
    
    # Tải dữ liệu
    X = preprocessor.X_scaled_df.values
    data_original = pd.read_csv(
        "data/Mall_Customers.csv"
    )[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    # Áp dụng PCA ngay sau khi scale để dữ liệu visualization là 2D
    preprocessor_pca = DataPreprocessor(use_pca=True, n_components=2)
    X_scaled_df, X_pca_df = preprocessor_pca.preprocess(
        file_path="data/Mall_Customers.csv", # Sử dụng lại file gốc để tiền xử lý lại với PCA
        output_path_scaled="data/processed_mall_customers.csv", # Giữ tên file cũ nếu cần
        output_path_pca="data/mall_customers_pca.csv"
    )
    X_pca = X_pca_df.values # Dữ liệu 2D cho visualization

    evaluator = ClusterEvaluator()
    visualizer = ClusterVisualizer()
    
    # K-Means Thư viện
    kmeans_lib = KMeansLibrary(k=5, random_state=42)
    # Fit trên dữ liệu gốc X (đã scale) để tính centroid trong không gian gốc
    labels_kmeans_lib, centroids_kmeans_lib = kmeans_lib.fit(X)
    # Chiếu centroids xuống 2D cho visualization
    centroids_lib_pca = preprocessor_pca.pca.transform(centroids_kmeans_lib)

    kmeans_lib.plot_elbow_method(X, max_k=10) # Elbow method dùng dữ liệu gốc
    evaluator.evaluate(X, labels_kmeans_lib, centroids=centroids_kmeans_lib, method_name="K-Means Library") # Đánh giá dùng dữ liệu gốc và centroids gốc
    visualizer.plot_clusters(
        X_pca, labels_kmeans_lib, centroids_lib_pca, # Plot dùng X_pca (2D) và centroids_lib_pca (2D)
        "K-Means Library", "results/kmeans_library_plot.png"
    )
    analyze_clusters(data_original, labels_kmeans_lib)
    
    # K-Means Tự viết
    kmeans_custom = KMeansCustom(k=5, random_state=42)
    # Fit trên dữ liệu gốc X (đã scale)
    labels_kmeans_custom, centroids_kmeans_custom = kmeans_custom.fit(X)
    # Chiếu centroids xuống 2D cho visualization
    centroids_custom_pca = preprocessor_pca.pca.transform(centroids_kmeans_custom)

    kmeans_custom.plot_elbow_method(X, max_k=10) # Elbow method dùng dữ liệu gốc
    evaluator.evaluate(X, labels_kmeans_custom, centroids=centroids_kmeans_custom, method_name="K-Means Custom") # Đánh giá dùng dữ liệu gốc và centroids gốc
    visualizer.plot_clusters(
        X_pca, labels_kmeans_custom, centroids_custom_pca, # Plot dùng X_pca (2D) và centroids_custom_pca (2D)
        "K-Means Custom", "results/kmeans_custom_plot.png"
    )
    
    # DBSCAN
    dbscan = DBSCANLibrary(eps=0.5, min_samples=5)
    # Fit trên dữ liệu gốc X (đã scale)
    labels_dbscan = dbscan.fit(X)
    num_clusters = len(np.unique(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    num_noise_points = sum(labels_dbscan == -1)
    print(f"DBSCAN: Số cụm = {num_clusters}, Số điểm nhiễu = {num_noise_points} / {len(labels_dbscan)}")

    dbscan.plot_k_distance(X, k=5) # k-distance dùng dữ liệu gốc
    evaluator.evaluate(X, labels_dbscan, centroids=None, method_name="DBSCAN") # Đánh giá dùng dữ liệu gốc, DBSCAN không có centroid
    visualizer.plot_clusters(
        X_pca, labels_dbscan, None, "DBSCAN", "results/dbscan_plot.png" # Plot dùng X_pca (2D), DBSCAN không có centroid 2D
    )
    
    # Lưu kết quả đánh giá
    evaluator.save_results()
    
    # Tạo báo cáo
    with open("results/report.txt", "w") as f:
        f.write("Clustering Project Report\n")
        f.write("=======================\n")
        f.write(f"Optimal k for K-Means: 5\n")
        f.write(f"DBSCAN Parameters: eps=0.5, min_samples=5\n")
        f.write("Evaluation Metrics:\n")
        for method, metrics in evaluator.results.items():
            f.write(f"{method}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nCluster Summary:\n")
        f.write(pd.read_csv("results/cluster_summary.csv").to_string())

if __name__ == "__main__":
    run_clustering()