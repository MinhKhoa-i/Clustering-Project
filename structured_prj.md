Gom nhóm dữ liệu bằng các phương pháp học máy

Clustering_Project/
├── data/                       # Thư mục chứa dữ liệu
│   ├── raw_data.csv           # Dữ liệu thô (ví dụ: Iris hoặc dữ liệu tự chọn)
│   └── processed_data.csv     # Dữ liệu sau khi tiền xử lý (nếu cần lưu)
├── src/                       # Thư mục chứa mã nguồn
│   ├── preprocessing.py       # Tiền xử lý dữ liệu
│   ├── kmeans_custom.py       # Giải thuật K-Means tự viết
│   ├── kmeans_library.py      # K-Means dùng thư viện scikit-learn
│   ├── dbscan_library.py      # DBSCAN dùng thư viện (hoặc tự viết nếu muốn)
│   ├── evaluation.py          # Hàm đánh giá (Silhouette, Davies-Bouldin, v.v.)
│   └── visualization.py       # Hàm trực quan hóa kết quả
├── notebooks/                 # Thư mục chứa Jupyter Notebook
│   └── main.ipynb             # File chính chạy toàn bộ quy trình
├── results/                   # Thư mục lưu kết quả
│   ├── kmeans_custom_plot.png # Biểu đồ K-Means tự viết
│   ├── kmeans_library_plot.png# Biểu đồ K-Means thư viện
│   ├── dbscan_plot.png        # Biểu đồ DBSCAN
│   └── evaluation_metrics.txt # Kết quả đánh giá số liệu
├── report/                    # Thư mục chứa báo cáo
│   ├── report.docx            # Báo cáo theo mẫu
│   └── figures/               # Hình ảnh minh họa trong báo cáo
│       ├── fig1_kmeans_custom.png
│       ├── fig2_kmeans_library.png
│       └── fig3_dbscan.png
├── app/                       # (Tùy chọn) Ứng dụng điểm cộng
│   ├── app.py                 # Ứng dụng đơn giản (ví dụ: Flask hoặc Streamlit)
│   └── templates/             # Giao diện nếu dùng Flask
│       └── index.html
├── requirements.txt           # Danh sách thư viện cần cài đặt
└── README.md                  # Hướng dẫn chạy dự án
