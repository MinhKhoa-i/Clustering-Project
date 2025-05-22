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
    
    def plot_evaluation_comparison(self, evaluation_results, metrics_to_compare, output_dir="results"):
        """
        Vẽ biểu đồ cột so sánh các độ đo đánh giá giữa các mô hình.

        Args:
            evaluation_results (dict): Từ điển chứa kết quả đánh giá, với key là tên phương pháp và value là các độ đo.
            metrics_to_compare (list): Danh sách các độ đo cần so sánh.
            output_dir (str): Thư mục để lưu biểu đồ.
        """
        methods = list(evaluation_results.keys())

        for metric in metrics_to_compare:
            # Thu thập giá trị của độ đo từ tất cả các phương pháp
            values = []
            valid_methods = []
            for method in methods:
                if metric in evaluation_results[method] and evaluation_results[method][metric] is not None:
                    values.append(evaluation_results[method][metric])
                    valid_methods.append(method)

            if not values:
                print(f"Không có dữ liệu để vẽ biểu đồ cho {metric}")
                continue

            # Vẽ biểu đồ cột
            plt.figure(figsize=(8, 6))
            bars = plt.bar(valid_methods, values, color=plt.cm.tab10(np.arange(len(valid_methods))))
            plt.title(f"So sánh {metric} giữa các phương pháp")
            plt.xlabel("Phương pháp phân cụm")
            plt.ylabel(metric)
            # Đặt giới hạn trục y tùy thuộc vào độ đo
            if metric == "Silhouette Score":
                plt.ylim(0, 0.5)
            elif metric == "Davies-Bouldin Index":
                plt.ylim(0, 2.0)
            elif metric == "Calinski-Harabasz Index":
                plt.ylim(0, 300)
            elif metric == "MSE":
                plt.ylim(0, 0.5)
            elif metric == "RMSE":
                plt.ylim(0, 0.7)
            # Thêm giá trị lên trên mỗi cột
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                         ha='center', va='bottom')
            # Xoay nhãn trục x để dễ đọc
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            # Lưu biểu đồ
            save_path = f"{output_dir}/{metric.lower().replace(' ', '_')}_comparison.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Biểu đồ so sánh {metric} đã lưu tại: {save_path}")

    def plot_model_evaluation_diagram(self, method_name, metrics, values, save_path):
        """
        Vẽ biểu đồ đánh giá cho một mô hình cụ thể, với trục x là các độ đo và trục y là giá trị.

        Args:
            method_name (str): Tên phương pháp (K-Means Library, K-Means Custom, DBSCAN).
            metrics (list): Danh sách các độ đo.
            values (list): Danh sách giá trị tương ứng với các độ đo.
            save_path (str): Đường dẫn lưu biểu đồ.
        """
        # Lọc bỏ các giá trị None và độ đo tương ứng
        valid_metrics = []
        valid_values = []
        for metric, value in zip(metrics, values):
            if value is not None:
                valid_metrics.append(metric)
                valid_values.append(value)

        if not valid_metrics:
            print(f"Không có dữ liệu để vẽ biểu đồ cho {method_name}")
            return

        # Vẽ biểu đồ cột
        plt.figure(figsize=(8, 6))
        bars = plt.bar(valid_metrics, valid_values, color=plt.cm.tab10(0))
        plt.title(f"Đánh giá mô hình {method_name} theo các độ đo")
        plt.xlabel("Độ đo đánh giá")
        plt.ylabel("Giá trị")
        # Đặt giới hạn trục y tùy thuộc vào giá trị lớn nhất
        max_value = max(valid_values)
        if max_value <= 0.5:
            plt.ylim(0, 0.5)
        elif max_value <= 2.0:
            plt.ylim(0, 2.0)
        else:
            plt.ylim(0, max_value * 1.1)
        # Thêm giá trị lên trên mỗi cột
        for bar, value in zip(bars, valid_values):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                     ha='center', va='bottom')
        # Xoay nhãn trục x để dễ đọc
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Lưu biểu đồ
        plt.savefig(save_path)
        plt.close()
        print(f"Sơ đồ đánh giá mô hình {method_name} đã lưu tại: {save_path}")