# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataPreprocessor:
    """Lớp tiền xử lý dữ liệu cho phân cụm."""
    
    def __init__(self, use_pca=True, n_components=2):
        """
        Khởi tạo bộ tiền xử lý.
        
        Args:
            use_pca (bool): Có dùng PCA không.
            n_components (int): Số chiều sau PCA.
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.X_scaled_df = None
        self.X_pca_df = None
    
    def preprocess(self, file_path, output_path_scaled="data/processed_data.csv", 
                   output_path_pca="data/processed_data_pca.csv"):
        """
        Tiền xử lý dữ liệu.
        
        Args:
            file_path (str): Đường dẫn file CSV đầu vào.
            output_path_scaled (str): Đường dẫn lưu dữ liệu chuẩn hóa.
            output_path_pca (str): Đường dẫn lưu dữ liệu PCA.
        
        Returns:
            tuple: (X_scaled_df, X_pca_df)
        """
        data = pd.read_csv(file_path)
        columns_to_drop = ["CustomerID"] if "CustomerID" in data.columns else []
        if "Gender" in data.columns:
            columns_to_drop.append("Gender")
        data = data.drop(columns=columns_to_drop, errors="ignore")
        
        if data.isnull().any().any():
            print("Cảnh báo: Dữ liệu có giá trị thiếu. Điền bằng giá trị trung bình.")
            data = data.fillna(data.mean())
        
        X = data.select_dtypes(include=["float64", "int64"]).values
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled_df = pd.DataFrame(X_scaled, columns=data.columns)
        
        self.X_scaled_df.to_csv(output_path_scaled, index=False)
        print(f"Dữ liệu đã chuẩn hóa được lưu tại: {output_path_scaled}")
        
        if self.use_pca:
            X_pca = self.pca.fit_transform(X_scaled)
            self.X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(self.n_components)])
            self.X_pca_df.to_csv(output_path_pca, index=False)
            print(f"Dữ liệu sau PCA được lưu tại: {output_path_pca}")
            print(f"Tỷ lệ phương sai được giải thích bởi PCA: {self.pca.explained_variance_ratio_}")
        
        return self.X_scaled_df, self.X_pca_df
    
    @staticmethod
    def load_processed_data(file_path):
        """
        Tải dữ liệu đã xử lý.
        
        Args:
            file_path (str): Đường dẫn file CSV.
        
        Returns:
            pd.DataFrame: Dữ liệu đã tải.
        """
        return pd.read_csv(file_path)

if __name__ == "__main__":
    preprocessor = DataPreprocessor(use_pca=True, n_components=2)
    preprocessor.preprocess(
        file_path="data/Mall_Customers.csv",
        output_path_scaled="data/processed_mall_customers.csv",
        output_path_pca="data/mall_customers_pca.csv"
    )