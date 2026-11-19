warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'requirements.txt', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/README.md b/README.md[m
[1mindex 555741c..2d560d2 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -1,2 +1,58 @@[m
[31m-# BTL_MachineLearning_NhanDangBienBao[m
[31m-Traffic Sign Recognition using Machine Learning (KNN, Decision Tree, HOG Feature Extraction)[m
[32m+[m[32m# Nhận dạng biển báo giao thông bằng Machine Learning[m
[32m+[m
[32m+[m[32m## 1. Mô tả[m
[32m+[m[32mDự án phân loại ảnh biển báo giao thông thuộc bộ dữ liệu GTSRB (43 lớp), sử dụng:[m
[32m+[m[32m- Tiền xử lý ảnh về kích thước 32x32, ảnh xám.[m
[32m+[m[32m- Hai cách biểu diễn: Pixel và HOG.[m
[32m+[m[32m- Các mô hình: KNN, Decision Tree và LinearSVC.[m
[32m+[m[32m- Đánh giá Accuracy, Precision, Recall, Macro F1, confusion matrix.[m
[32m+[m[32m- Lưu mô hình để nhận dạng ảnh mới; giao diện Streamlit.[m
[32m+[m
[32m+[m[32m## 2. Cài đặt (Windows PowerShell)[m
[32m+[m[32m```powershell[m
[32m+[m[32mcd traffic_sign_recognition[m
[32m+[m[32mpy -m venv .venv[m
[32m+[m[32m.\.venv\Scripts\Activate.ps1[m
[32m+[m[32mpython -m pip install --upgrade pip[m
[32m+[m[32mpip install -r requirements.txt[m
[32m+[m[32m```[m
[32m+[m[32mNếu PowerShell chặn kích hoạt môi trường:[m
[32m+[m[32m```powershell[m
[32m+[m[32mSet-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass[m
[32m+[m[32m.\.venv\Scripts\Activate.ps1[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m## 3. Dữ liệu[m
[32m+[m[32mMặc định chương trình tải GTSRB bằng `torchvision.datasets.GTSRB` vào thư mục `data/`.[m
[32m+[m[32mLần chạy đầu cần Internet và có thể mất thời gian. Nếu thư viện torchvision trên máy không[m
[32m+[m[32mtương thích, cài bản PyTorch/torchvision phù hợp với hệ điều hành tại trang chính thức PyTorch.[m
[32m+[m
[32m+[m[32m## 4. Chạy huấn luyện[m
[32m+[m[32m```powershell[m
[32m+[m[32mpython main.py --download --max-train 0 --max-test 0[m
[32m+[m[32m```[m
[32m+[m[32m`0` nghĩa là dùng toàn bộ dữ liệu. Để chạy thử nhanh trước:[m
[32m+[m[32m```powershell[m
[32m+[m[32mpython main.py --download --max-train 5000 --max-test 2000[m
[32m+[m[32m```[m
[32m+[m[32mKết quả nằm trong `outputs/`; mô hình tốt nhất nằm trong `models/`.[m
[32m+[m
[32m+[m[32m## 5. Chạy giao diện[m
[32m+[m[32mSau khi huấn luyện xong:[m
[32m+[m[32m```powershell[m
[32m+[m[32mstreamlit run app.py[m
[32m+[m[32m```[m
[32m+[m[32mTải lên một ảnh biển báo để xem lớp dự đoán và xác suất (LinearSVC dùng hiệu chỉnh xác suất[m
[32m+[m[32mCalibratedClassifierCV nếu có; các mô hình khác dùng predict_proba).[m
[32m+[m
[32m+[m[32m## 6. Cấu trúc[m
[32m+[m[32m- `main.py`: điều phối tải dữ liệu, huấn luyện và đánh giá.[m
[32m+[m[32m- `src/data.py`: nạp GTSRB và tiền xử lý.[m
[32m+[m[32m- `src/features.py`: Pixel/HOG.[m
[32m+[m[32m- `src/models.py`: định nghĩa mô hình.[m
[32m+[m[32m- `src/evaluate.py`: chỉ số và biểu đồ.[m
[32m+[m[32m- `app.py`: giao diện dự đoán ảnh mới.[m
[32m+[m
[32m+[m[32m## Lưu ý học thuật[m
[32m+[m[32mChỉ ghi vào báo cáo các chỉ số thực tế được sinh ra trong `outputs/metrics.csv`.[m
[32m+[m[32mKhông sử dụng số liệu tham khảo trên Internet như thể là kết quả tự chạy của nhóm.[m
[1mdiff --git a/requirements.txt b/requirements.txt[m
[1mindex e69de29..23abb75 100644[m
[1m--- a/requirements.txt[m
[1m+++ b/requirements.txt[m
[36m@@ -0,0 +1,11 @@[m
[32m+[m[32mnumpy>=1.24[m
[32m+[m[32mpandas>=2.0[m
[32m+[m[32mmatplotlib>=3.7[m
[32m+[m[32mseaborn>=0.12[m
[32m+[m[32mscikit-learn>=1.3[m
[32m+[m[32mscikit-image>=0.21[m
[32m+[m[32mjoblib>=1.3[m
[32m+[m[32mPillow>=10.0[m
[32m+[m[32mstreamlit>=1.30[m
[32m+[m[32mtorchvision>=0.16[m
[32m+[m[32mtorch>=2.1[m
