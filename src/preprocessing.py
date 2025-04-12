import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(file_path, output_path_scaled="data/processed_data.csv", 
                    output_path_pca="data/processed_data_pca.csv", use_pca=True, n_components=2):
    """
    Tiền xử lý dữ liệu cho bài toán phân cụm.
    
    Args:
        file_path (str): Đường dẫn đến file CSV đầu vào (ví dụ: Mall_Customers.csv).
        output_path_scaled (str): Đường dẫn lưu dữ liệu đã chuẩn hóa.
        output_path_pca (str): Đường dẫn lưu dữ liệu sau PCA (nếu dùng).
        use_pca (bool): Có giảm chiều bằng PCA hay không.
        n_components (int): Số chiều sau PCA (mặc định là 2).
    
    Returns:
        tuple: (X_scaled_df, X_pca_df) - Dữ liệu đã chuẩn hóa và dữ liệu sau PCA (nếu có).
    """
    # Đọc dữ liệu
    data = pd.read_csv(file_path)
    
    # Loại bỏ cột không cần thiết
    # Giả sử dữ liệu có các cột như Mall Customers: CustomerID, Gender, Age, Annual Income, Spending Score
    columns_to_drop = ["CustomerID"] if "CustomerID" in data.columns else []
    if "Gender" in data.columns:
        columns_to_drop.append("Gender")  # Loại Gender để dùng đặc trưng số
    data = data.drop(columns=columns_to_drop, errors="ignore")
    
    # Kiểm tra giá trị thiếu
    if data.isnull().any().any():
        print("Cảnh báo: Dữ liệu có giá trị thiếu. Điền bằng giá trị trung bình.")
        data = data.fillna(data.mean())
    
    # Chọn tất cả các cột số còn lại
    X = data.select_dtypes(include=["float64", "int64"]).values
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chuyển lại thành DataFrame để dễ quản lý
    X_scaled_df = pd.DataFrame(X_scaled, columns=data.columns)
    
    # Lưu dữ liệu đã chuẩn hóa
    X_scaled_df.to_csv(output_path_scaled, index=False)
    print(f"Dữ liệu đã chuẩn hóa được lưu tại: {output_path_scaled}")
    
    # Giảm chiều bằng PCA (nếu yêu cầu)
    X_pca_df = None
    if use_pca:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
        X_pca_df.to_csv(output_path_pca, index=False)
        print(f"Dữ liệu sau PCA được lưu tại: {output_path_pca}")
        print(f"Tỷ lệ phương sai được giải thích bởi PCA: {pca.explained_variance_ratio_}")
    
    return X_scaled_df, X_pca_df

def load_processed_data(file_path):
    """
    Tải dữ liệu đã tiền xử lý.
    
    Args:
        file_path (str): Đường dẫn đến file CSV đã xử lý.
    
    Returns:
        pd.DataFrame: Dữ liệu đã tải.
    """
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Ví dụ chạy tiền xử lý với Mall Customer Segmentation Data
    input_file = "data/Mall_Customers.csv"
    preprocess_data(
        file_path=input_file,
        output_path_scaled="data/processed_mall_customers.csv",
        output_path_pca="data/mall_customers_pca.csv",
        use_pca=True,
        n_components=2
    )