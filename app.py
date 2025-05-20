import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.kmeans_library import KMeansLibrary
from src.kmeans_custom import KMeansCustom
from src.dbscan_library import DBSCANLibrary
from src.preprocessing import DataPreprocessor
from src.visualization import ClusterVisualizer
from src.evaluation import ClusterEvaluator
from sklearn.neighbors import NearestNeighbors  

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân Tích Phân Cụm", layout="wide")

# Tiêu đề ứng dụng
st.title("Phân Tích Phân Cụm Khách Hàng Trung Tâm Thương Mại")

# Sidebar để chọn thuật toán và tham số
st.sidebar.header("Cài Đặt Thuật Toán")
algorithm = st.sidebar.selectbox(
    "Chọn Thuật Toán Phân Cụm",
    ["K-Means (Thư viện)", "K-Means (Tự viết)", "DBSCAN"]
)

# Tham số cho thuật toán
if algorithm.startswith("K-Means"):
    k = st.sidebar.slider("Số lượng cụm (k)", 2, 10, 5)
    random_state = 42
else:  # DBSCAN
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, step=0.1)
    min_samples = st.sidebar.slider("Số điểm tối thiểu", 2, 20, 5)

# Tiền xử lý dữ liệu
@st.cache_data
def preprocess_data():
    preprocessor = DataPreprocessor(use_pca=True, n_components=2)
    preprocessor.preprocess(
        file_path="data/Mall_Customers.csv",
        output_path_scaled="data/processed_mall_customers.csv",
        output_path_pca="data/mall_customers_pca.csv"
    )
    return preprocessor.X_scaled_df, preprocessor.X_pca_df

X_scaled_df, X_pca_df = preprocess_data()
X = X_scaled_df.values
X_pca = X_pca_df.values
data_original = pd.read_csv("data/Mall_Customers.csv")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
]

# Chạy thuật toán phân cụm
visualizer = ClusterVisualizer()
evaluator = ClusterEvaluator()

if algorithm == "K-Means (Thư viện)":
    model = KMeansLibrary(k=k, random_state=random_state)
    labels, centroids = model.fit(X)
    method_name = "K-Means Thư viện"
elif algorithm == "K-Means (Tự viết)":
    model = KMeansCustom(k=k, random_state=random_state)
    labels, centroids = model.fit(X)
    method_name = "K-Means Tự viết"
else:
    model = DBSCANLibrary(eps=eps, min_samples=min_samples)
    labels = model.fit(X)
    centroids = None
    method_name = "DBSCAN"

# Đánh giá phân cụm
metrics = evaluator.evaluate(X, labels, method_name)

# Hiển thị kết quả
st.header("Kết Quả Phân Cụm")

# Biểu đồ phân cụm
st.subheader("Trực Quan Hóa Cụm")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
if centroids is not None:
    ax.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200,
               linewidths=3, label="Tâm cụm")
ax.set_title(f"Kết Quả Phân Cụm - {method_name}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
st.pyplot(fig)

# Biểu đồ Elbow hoặc k-Distance
st.subheader("Lựa Chọn Tham Số")
if algorithm.startswith("K-Means"):
    inertias = []
    for k in range(1, 11):
        model_k = KMeansLibrary(k, random_state) if algorithm == "K-Means (Thư viện)" else KMeansCustom(k, random_state)
        model_k.fit(X)
        inertias.append(model_k.compute_inertia(X))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, 11), inertias, marker='o')
    ax.set_title("Phương Pháp Elbow để Chọn k Tối Ưu")
    ax.set_xlabel("Số lượng cụm (k)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)
else:
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, 4], axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(distances)
    ax.set_title("Biểu Đồ k-Distance cho DBSCAN")
    ax.set_xlabel("Điểm được sắp xếp theo khoảng cách")
    ax.set_ylabel("Khoảng cách hàng xóm thứ 5")
    st.pyplot(fig)

# Chỉ số đánh giá
st.subheader("Chỉ Số Đánh Giá")
if metrics:
    st.write(f"Silhouette Score: {metrics['Silhouette Score']:.4f}")
    st.write(f"Davies-Bouldin Index: {metrics['Davies-Bouldin Index']:.4f}")
else:
    st.write("Không đủ cụm để đánh giá.")

# Tóm tắt cụm
st.subheader("Tóm Tắt Cụm")
data_original["Cluster"] = labels
cluster_summary = data_original.groupby("Cluster").mean()
st.write(cluster_summary)

# Lưu kết quả
if st.button("Lưu Kết Quả"):
    evaluator.save_results("results/evaluation_metrics.txt")
    cluster_summary.to_csv("results/cluster_summary.csv")
    st.success("Kết quả đã được lưu thành công!")