# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import from src
from src.kmeans_custom import KMeansCustom
from src.dbscan_library import DBSCANLibrary
from src.evaluation import ClusterEvaluator
from src.visualization import ClusterVisualizer

# Hàm vẽ ranh giới quyết định
def plot_decision_boundary(ax, xx, yy, labels_grid, cmap):
    """Vẽ ranh giới quyết định trên lưới."""
    ax.contourf(xx, yy, labels_grid.reshape(xx.shape), alpha=0.3, cmap=cmap)

# Khởi tạo session state để lưu trữ kết quả đánh giá
if 'eval_history' not in st.session_state:
    st.session_state.eval_history = {}

st.title("Phân cụm bằng K-Means hoặc DBSCAN")

# 1. Upload CSV file
uploaded = st.sidebar.file_uploader("Upload CSV for clustering", type=["csv"])
if not uploaded:
    st.sidebar.warning("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("**Uploaded Data Preview**")
st.write(df.head())

# 2. Feature selection
cols = st.sidebar.multiselect(
    "Select columns for clustering", df.columns.tolist(), default=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
)
if len(cols) < 2:
    st.sidebar.warning("Please select at least 2 columns.")
    st.stop()

# 2.1 Handle features: Numeric and categorical via one-hot encoding
df_selected = df[cols]
df_processed = pd.get_dummies(df_selected, drop_first=True)
X_raw = df_processed.values
X = StandardScaler().fit_transform(X_raw)
st.write(f"**Selected Features (after preprocessing)**: {df_processed.columns.tolist()}")

# 3. Algorithm options
method = st.sidebar.selectbox("Clustering Method", ["K-Means", "DBSCAN"])
if method == "K-Means":
    algo = st.sidebar.selectbox("Algorithm", ["Custom", "Scikit-learn"])
    k = st.sidebar.slider("K clusters", 2, 10, 5)
    max_iters = st.sidebar.number_input("Max iterations", 10, 1000, 300)
    tol = st.sidebar.number_input("Tolerance", 1e-6, 1e-1, 1e-4, format="%.1e")
    random_state = st.sidebar.number_input("Random state", 0, 9999, 42)
else:  # DBSCAN
    algo = "DBSCAN"
    eps = st.sidebar.number_input("Epsilon (eps)", 0.1, 2.0, 0.5, step=0.1)
    min_samples = st.sidebar.number_input("Min samples", 2, 20, 5)
    random_state = None  # DBSCAN không cần random_state

# Danh sách các độ đo để vẽ biểu đồ (chỉ lấy Silhouette Score và Davies-Bouldin Index)
metrics_to_compare = [
    "Silhouette Score",
    "Davies-Bouldin Index"
]

if st.sidebar.button("Run Clustering"):
    # 4. Run clustering
    if method == "K-Means":
        if algo == "Custom":
            km = KMeansCustom(k=k, max_iters=max_iters, tol=tol, random_state=random_state)
            labels, centers = km.fit(X)
            method_name = "K-Means (Custom)"
        else:
            km = KMeans(n_clusters=k, max_iter=max_iters, tol=tol,
                        random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            centers = km.cluster_centers_
            method_name = "K-Means (Scikit-learn)"
    else:  # DBSCAN
        dbscan = DBSCANLibrary(eps=eps, min_samples=min_samples)
        labels = dbscan.fit(X)
        centers = None  # DBSCAN không có tâm cụm
        method_name = f"DBSCAN (eps={eps}, min_samples={min_samples})"

    # 5. Evaluation using ClusterEvaluator
    evaluator = ClusterEvaluator()
    eval_results = evaluator.evaluate(X, labels, centroids=centers, method_name=method_name)
    
    # Lưu kết quả đánh giá vào session state
    st.session_state.eval_history[method_name] = eval_results

    st.subheader("📊 Evaluation Metrics")
    st.table(pd.DataFrame.from_dict(eval_results, orient='index', columns=['Value']))

    # 6. Visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        centers2 = pca.transform(centers) if centers is not None else None
        st.write(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    else:
        X2, centers2 = X, centers

    # Create meshgrid for decision boundary (chỉ áp dụng cho K-Means)
    if method == "K-Means":
        x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
        y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_orig = pca.inverse_transform(grid) if X.shape[1] > 2 else grid

        if algo == "Custom":
            dist_grid = np.linalg.norm(grid_orig[:, None, :] - centers[None, :, :], axis=2)
            labels_grid = np.argmin(dist_grid, axis=1)
        else:
            labels_grid = km.predict(grid_orig)
    else:
        # DBSCAN không có tâm cụm, không vẽ ranh giới quyết định
        xx, yy, labels_grid = None, None, None

    # Plot clustering results (chỉ giữ 1 biểu đồ)
    fig, ax = plt.subplots(figsize=(8, 6))  # Chỉ tạo 1 subplot
    if method == "K-Means":
        cmap = plt.cm.get_cmap('tab10', k)
    else:
        # DBSCAN có thể có nhãn -1 (nhiễu), cần tính số cụm thực tế
        unique_labels = np.unique(labels)
        k = len(unique_labels) - (1 if -1 in unique_labels else 0)
        cmap = plt.cm.get_cmap('tab10', max(k, 1))

    # Decision Boundary (chỉ cho K-Means)
    if method == "K-Means":
        plot_decision_boundary(ax, xx, yy, labels_grid, cmap)
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, edgecolor='k', s=30, cmap=cmap)
        ax.scatter(centers2[:, 0], centers2[:, 1], marker='X', s=200, c='red', label='Centroids')
        ax.set_title("Clustering Results with Decision Boundary")
        patches = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(k)]
        legend = ax.legend(handles=patches + [mpatches.Patch(color='red', label='Centroids')],
                           loc='best')
        ax.add_artist(legend)
    else:
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, edgecolor='k', s=30, cmap=cmap)
        ax.set_title("DBSCAN Clustering Results")
        unique_labels = np.unique(labels)
        patches = []
        for label in unique_labels:
            if label == -1:
                patches.append(mpatches.Patch(color='black', label='Noise'))
            else:
                patches.append(mpatches.Patch(color=cmap(label), label=f'Cluster {label}'))
        legend = ax.legend(handles=patches, loc='best')
        ax.add_artist(legend)
    ax.set_xlabel("Feature 1 (PCA)")
    ax.set_ylabel("Feature 2 (PCA)")

    st.pyplot(fig)

    # 7. User explanation
    explanation = """
    **Giải thích biểu đồ**:
    """
    if method == "K-Means":
        explanation += """
    - **Vùng màu**: Mỗi vùng biểu thị khu vực không gian mà tất cả các điểm trong đó sẽ được gán vào cùng một cụm.
      - Vùng **lớn nhất** cho thấy cluster đó có phạm vi phân bố rộng và centroid cách xa các cluster khác, nghĩa là mô hình gán nhiều không gian cho cluster này.
      - Vùng **nhỏ nhất** cho thấy cluster tập trung chặt chẽ quanh centroid, chiếm ít không gian hơn.
    - **Điểm màu**: Vị trí các mẫu dữ liệu, màu sắc thể hiện cụm đã được gán; mật độ điểm trong mỗi vùng còn phản ánh độ dày đặc (density) của dữ liệu.
    - **Chữ X đỏ**: Vị trí trung tâm (centroid) của từng cụm; đây là điểm trung bình của tất cả các dữ liệu trong cluster.
    - **Ý nghĩa thực tiễn**:
      + Cluster có **vùng lớn** thường bao quát các khách hàng với hành vi/phân phối đa dạng hơn (ví dụ: nhóm khách hàng có thu nhập và chi tiêu trung bình).
      + Cluster có **vùng nhỏ** thường đại diện cho nhóm khách hàng có tính chất đồng nhất và tập trung (ví dụ: nhóm khách hàng trẻ chi tiêu cao).
    """
    else:
        explanation += """
    - **Điểm màu**: Vị trí các mẫu dữ liệu, màu sắc thể hiện cụm đã được gán. Điểm màu đen (nếu có) biểu thị nhiễu (noise), tức là các điểm không thuộc cụm nào.
    - **Không có ranh giới quyết định**: DBSCAN không tạo tâm cụm, nên không thể vẽ ranh giới như K-Means.
    - **Ý nghĩa thực tiễn**:
      + Các cụm được tạo bởi DBSCAN thường có hình dạng bất kỳ, phù hợp để phát hiện các nhóm khách hàng có hành vi đặc biệt.
      + Các điểm nhiễu (noise) có thể là các khách hàng không thuộc nhóm nào, cần phân tích thêm để hiểu nguyên nhân.
    """
    st.markdown(explanation)

    # 8. Cluster summary
    st.subheader("📋 Cluster Summary")
    df["Cluster"] = labels
    cluster_summary = df.groupby("Cluster")[cols].mean()
    st.table(cluster_summary)

    # 9. Additional information for DBSCAN
    if method == "DBSCAN":
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise = np.sum(labels == -1)
        st.write(f"**Number of Clusters**: {num_clusters}")
        st.write(f"**Number of Noise Points**: {num_noise} (out of {len(labels)} points)")

    # 10. Hiển thị sơ đồ đánh giá mô hình được chọn
    st.subheader(f"📈 Biểu đồ đánh giá mô hình {method_name}")
    visualizer = ClusterVisualizer()
    values = [eval_results.get(metric, None) for metric in metrics_to_compare]
    # Vẽ biểu đồ và hiển thị trong Streamlit
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_metrics = []
    valid_values = []
    for metric, value in zip(metrics_to_compare, values):
        if value is not None:
            valid_metrics.append(metric)
            valid_values.append(value)

    if valid_metrics:
        bars = ax.bar(valid_metrics, valid_values, color=plt.cm.tab10(0))
        ax.set_title(f"Đánh giá mô hình {method_name} theo các độ đo")
        ax.set_xlabel("Độ đo đánh giá")
        ax.set_ylabel("Giá trị")
        # Đặt giới hạn trục y tùy thuộc vào giá trị lớn nhất
        max_value = max(valid_values)
        if max_value <= 0.5:
            ax.set_ylim(0, 0.5)
        elif max_value <= 2.0:
            ax.set_ylim(0, 2.0)
        else:
            ax.set_ylim(0, max_value * 1.1)
        # Thêm giá trị lên trên mỗi cột
        for bar, value in zip(bars, valid_values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                    ha='center', va='bottom')
        # Xoay nhãn trục x để dễ đọc
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Không có dữ liệu để vẽ sơ đồ đánh giá cho mô hình này.")