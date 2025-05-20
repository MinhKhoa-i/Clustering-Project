# Clustering Project

## Giới thiệu

Dự án này thực hiện phân tích phân cụm (Clustering Analysis) trên tập dữ liệu khách hàng sử dụng các thuật toán K-Means (tự cài đặt và sử dụng thư viện Scikit-learn) và DBSCAN (sử dụng thư viện Scikit-learn). Dự án cũng bao gồm một ứng dụng Streamlit cho phép người dùng tương tác để thực hiện phân cụm và trực quan hóa kết quả.

## Cấu trúc thư mục

- `data/`: Chứa tập dữ liệu đầu vào (`Mall_Customers.csv`) và dữ liệu đã tiền xử lý.
- `report/`: Dự kiến chứa báo cáo cuối cùng của dự án.
- `results/`: Chứa các kết quả đầu ra như biểu đồ, file tóm tắt cụm và báo cáo đánh giá.
- `src/`: Chứa mã nguồn các module:
    - `preprocessing.py`: Tiền xử lý dữ liệu (chuẩn hóa, PCA).
    - `kmeans_custom.py`: Triển khai thuật toán K-Means tự viết.
    - `kmeans_library.py`: Triển khai thuật toán K-Means sử dụng Scikit-learn.
    - `dbscan_library.py`: Triển khai thuật toán DBSCAN sử dụng Scikit-learn.
    - `evaluation.py`: Các hàm đánh giá chất lượng phân cụm.
    - `visualization.py`: Các hàm trực quan hóa kết quả.
    - `run_clustering.py`: Script chạy toàn bộ quy trình (tiền xử lý, phân cụm 3 cách, đánh giá, trực quan hóa).
- `app.py`: Mã nguồn ứng dụng Streamlit tương tác.
- `requirements.txt`: Danh sách các thư viện Python cần thiết.

## Cài đặt

Để chạy dự án, bạn cần cài đặt Python và các thư viện cần thiết. Sử dụng `pip`:

```bash
pip install -r requirements.txt
```

## Cách chạy

### Chạy từng module trong `src/`

Các file trong `src/` có thể chạy độc lập để kiểm tra hoặc sử dụng từng phần. Bạn có thể truyền đường dẫn file CSV làm đối số dòng lệnh (nếu không truyền, sẽ sử dụng file mặc định):

- **Tiền xử lý dữ liệu (`src/preprocessing.py`)**:
  ```bash
  python src/preprocessing.py [đường_dẫn_csv_đầu_vào] [đường_dẫn_csv_scaled] [đường_dẫn_csv_pca]
  # Ví dụ mặc định:
  # python src/preprocessing.py data/Mall_Customers.csv data/processed_mall_customers.csv data/mall_customers_pca.csv
  ```

- **KMeans Tự viết (`src/kmeans_custom.py`)**:
  ```bash
  python src/kmeans_custom.py [đường_dẫn_csv_scaled]
  # Ví dụ mặc định:
  # python src/kmeans_custom.py data/processed_mall_customers.csv
  ```

- **KMeans Thư viện (`src/kmeans_library.py`)**:
  ```bash
  python src/kmeans_library.py [đường_dẫn_csv_scaled]
  # Ví dụ mặc định:
  # python src/kmeans_library.py data/processed_mall_customers.csv
  ```

- **DBSCAN Thư viện (`src/dbscan_library.py`)**:
  ```bash
  python src/dbscan_library.py [đường_dẫn_csv_scaled]
  # Ví dụ mặc định:
  # python src/dbscan_library.py data/processed_mall_customers.csv
  ```

- **Chạy toàn bộ quy trình (`src/run_clustering.py`)**:
  ```bash
  python src/run_clustering.py
  ```
  Lệnh này sẽ chạy tuần tự tiền xử lý, 3 phương pháp phân cụm, đánh giá và tạo các file kết quả, biểu đồ trong thư mục `results/`.

### Chạy ứng dụng Streamlit (`app.py`)

Ứng dụng này cung cấp giao diện tương tác để tải dữ liệu, chọn cột, chạy thuật toán (KMeans Tự viết, KMeans Thư viện, DBSCAN) và xem kết quả trực tiếp. Chạy ứng dụng bằng lệnh:

```bash
streamlit run app.py
```
Ứng dụng sẽ mở trên trình duyệt web của bạn (thường tại `http://localhost:8501`).

## Kết quả

Sau khi chạy `src/run_clustering.py` hoặc sử dụng ứng dụng Streamlit, các kết quả trực quan hóa (biểu đồ `.png`) và file tóm tắt, đánh giá sẽ được lưu trong thư mục `results/`.
