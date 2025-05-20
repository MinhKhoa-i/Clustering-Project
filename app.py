import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_decision_boundary(ax, xx, yy, labels_grid, cmap):
    """
    Vẽ ranh giới quyết định cho các cụm.
    Args:
        ax (matplotlib.axes.Axes): Axes để vẽ.
        xx (np.ndarray): Lưới x.
        yy (np.ndarray): Lưới y.
        labels_grid (np.ndarray): Nhãn cho lưới.
        cmap (matplotlib.colors.Colormap): Bản đồ màu.
    """
    ax.contourf(xx, yy, labels_grid.reshape(xx.shape), alpha=0.3, cmap=cmap)

# import functions from utils
from src.kmeans_custom import KMeansCustom
from src.kmeans_library import KMeansLibrary
from src.evaluation import ClusterEvaluator
from src.dbscan_library import DBSCANLibrary

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân Tích Phân Cụm", layout="wide")

# Tiêu đề ứng dụng
st.title("Ứng dụng phân cụm - Modular K-Means")

# 1. Load data
uploaded = st.sidebar.file_uploader("Tải lên file CSV", type=["csv"])
if not uploaded:
    st.sidebar.warning("Vui lòng tải lên một file CSV để tiếp tục.")
    st.stop()

df = pd.read_csv(uploaded)

# 2. Feature selection
cols = st.sidebar.multiselect(
    "Chọn các cột để phân cụm", df.columns.tolist(), df.columns[:2].tolist()
)
if len(cols) < 2:
    st.sidebar.warning("Vui lòng chọn ít nhất 2 cột.")
    st.stop()

# 2.1 Handle features
# Numeric and categorical via one-hot
df_selected = df[cols]
df_processed = pd.get_dummies(df_selected, drop_first=True)
X_raw = df_processed.values
X = StandardScaler().fit_transform(X_raw)

# 3. Algorithm options
algo = st.sidebar.selectbox("Thuật toán", ["KMeans (Viết tay)", "KMeans (Thư viện)", "DBSCAN"])
k = st.sidebar.slider("Số cụm K", 2, 10, 3)
max_iter = st.sidebar.number_input("Số vòng lặp tối đa", 10, 1000, 300)
tol = st.sidebar.number_input("Sai số (Tolerance)", 1e-6, 1e-1, 1e-4, format="%.1e")
random_state = st.sidebar.number_input("Trạng thái ngẫu nhiên (Random state)", 0, 9999, 42)
if algo == "DBSCAN":
    eps = st.sidebar.number_input("Bán kính (eps)", 0.01, 5.0, 0.5, format="%.2f")
    min_samples = st.sidebar.number_input("Số mẫu tối thiểu (min_samples)", 1, 20, 5)

if st.sidebar.button("Chạy phân cụm"):
    # 4. Run clustering
    if algo == "KMeans (Viết tay)":
        kmeans = KMeansCustom(k=k, max_iters=max_iter, random_state=random_state)
        labels, centers = kmeans.fit(X)
    elif algo == "KMeans (Thư viện)":
        kmeans = KMeansLibrary(k=k, random_state=random_state)
        labels, centers = kmeans.fit(X)
    else:  # DBSCAN
        dbscan = DBSCANLibrary(eps=eps, min_samples=min_samples)
        labels = dbscan.fit(X)
        centers = None

    # 5. Evaluation
    evaluator = ClusterEvaluator()
    scores = evaluator.evaluate(X, labels, centers, method_name=algo)
    st.subheader("📊 Các chỉ số đánh giá")
    st.table(pd.DataFrame.from_dict(scores, orient='index', columns=['Giá trị']))

    # Giải thích các chỉ số đánh giá
    st.markdown("""
    **Giải thích các chỉ số đánh giá**:

    - **Silhouette Score**: Đo lường mức độ phù hợp của điểm dữ liệu với cụm của nó và sự tách biệt với cụm khác. Giá trị từ -1 đến 1. **Gần 1 là tốt nhất** (cụm rõ ràng), gần 0 là chồng lấn, gần -1 là sai cụm.
    - **Davies-Bouldin Index**: Đo lường tỷ lệ trung bình của độ tương đồng giữa các cụm. Giá trị không âm. **Càng thấp càng tốt** (các cụm tách biệt và tập trung).
    - **Calinski-Harabasz Index**: Đo lường tỷ lệ phương sai giữa các cụm so với bên trong cụm. Giá trị cao. **Càng cao càng tốt** (cụm rõ ràng, gắn kết).
    - **MSE (Mean Squared Error)** & **RMSE (Root Mean Squared Error)**: (Chỉ cho K-Means) Đo lường tổng bình phương/căn bậc hai bình phương khoảng cách từ điểm đến tâm cụm. **Càng thấp càng tốt**.
    """)

    # 7. Cluster Summary
    st.subheader("📋 Tóm tắt các cụm")

    # Tạo DataFrame tạm với nhãn cụm
    df_clustered = df_selected.copy()
    df_clustered['Cụm'] = labels

    # Tính số lượng điểm trong mỗi cụm
    cluster_counts = df_clustered['Cụm'].value_counts().sort_index()
    st.write("**Số lượng điểm trong mỗi cụm:**")
    st.table(cluster_counts)

    # Tính đặc điểm trung bình của mỗi cụm
    cluster_summary = df_clustered.groupby('Cụm').mean()
    st.write("**Đặc điểm trung bình của mỗi cụm:**")
    st.table(cluster_summary)

    # 6. Visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        centers2 = pca.transform(centers) if centers is not None else None
    else:
        X2, centers2 = X, centers

    # meshgrid
    x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
    y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_orig = pca.inverse_transform(grid) if X.shape[1] > 2 else grid

    if algo.startswith("KMeans"):
        if centers is not None:
            dist_grid = np.linalg.norm(grid_orig[:,None,:] - centers[None,:,:], axis=2)
            labels_grid = np.argmin(dist_grid, axis=1)
        else:
            labels_grid = np.zeros(len(grid_orig), dtype=int)
    else:
        # DBSCAN không có decision boundary rõ ràng, gán -1 cho tất cả
        labels_grid = np.full(len(grid_orig), -1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))
    # Decision Boundary
    plot_decision_boundary(ax1, xx, yy, labels_grid, cmap)
    ax1.scatter(X2[:,0], X2[:,1], c=labels, edgecolor='k', s=30, cmap=cmap)
    if centers2 is not None:
        ax1.scatter(centers2[:,0], centers2[:,1], marker='X', s=200, c='red')
    ax1.set_title("Ranh giới quyết định")
    # Cluster Scatter with Boundary
    plot_decision_boundary(ax2, xx, yy, labels_grid, cmap)
    ax2.scatter(X2[:,0], X2[:,1], c=labels, edgecolor='k', s=30, cmap=cmap)
    if centers2 is not None:
        ax2.scatter(centers2[:,0], centers2[:,1], marker='X', s=200, c='red')
    ax2.set_title("Phân bố cụm và ranh giới")
    st.pyplot(fig)

    # 7. User explanation
    explanation = """
    **Giải thích biểu đồ**:
    - **Vùng màu**: mỗi vùng biểu thị khu vực không gian mà tất cả các điểm trong đó sẽ được gán vào cùng một cụm.
      - Vùng **lớn nhất** cho thấy cluster đó có phạm vi phân bố rộng và centroid cách xa các cluster khác, nghĩa là mô hình gán nhiều không gian cho cluster này.
      - Vùng **nhỏ nhất** cho thấy cluster tập trung chặt chẽ quanh centroid, chiếm ít không gian hơn.
    - **Điểm màu**: vị trí các mẫu dữ liệu, màu sắc thể hiện cụm đã được gán; mật độ điểm trong mỗi vùng còn phản ánh độ dày đặc (density) của dữ liệu.
    - **Chữ X đỏ**: vị trí trung tâm (centroid) của từng cụm; đây là điểm trung bình của tất cả các dữ liệu trong cluster.
    - **Ý nghĩa thực tiễn**:
      + Cluster có **vùng lớn** thường bao quát các khách hàng (hoặc mẫu) với hành vi/phân phối đa dạng hơn.
      + Cluster có **vùng nhỏ** thường đại diện cho nhóm mẫu có tính chất đồng nhất và tập trung.
    """
    st.markdown(explanation)

def plot_clusters(ax, X, labels, centroids=None, method_name="", cmap_name="tab10"):
    import matplotlib.patches as mpatches
    import numpy as np
    k = len(np.unique(labels))
    cmap = plt.get_cmap(cmap_name, k)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=50, alpha=0.8, edgecolor='k', linewidth=0.5)
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=250, linewidths=3, edgecolor='black', label='Centroids')
        for i, (x, y) in enumerate(centroids):
            ax.text(x, y, f'C{i}', fontsize=14, fontweight='bold', color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_title(method_name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle='--', alpha=0.3)
    handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(k)]
    handles.append(mpatches.Patch(color='red', label='Centroids'))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)