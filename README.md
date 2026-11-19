# Nhận dạng biển báo giao thông bằng Machine Learning

## 1. Mô tả
Dự án phân loại ảnh biển báo giao thông thuộc bộ dữ liệu GTSRB (43 lớp), sử dụng:
- Tiền xử lý ảnh về kích thước 32x32, ảnh xám.
- Hai cách biểu diễn: Pixel và HOG.
- Các mô hình: KNN, Decision Tree và LinearSVC.
- Đánh giá Accuracy, Precision, Recall, Macro F1, confusion matrix.
- Lưu mô hình để nhận dạng ảnh mới; giao diện Streamlit.

## 2. Cài đặt (Windows PowerShell)
```powershell
cd traffic_sign_recognition
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Nếu PowerShell chặn kích hoạt môi trường:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Dữ liệu
Mặc định chương trình tải GTSRB bằng `torchvision.datasets.GTSRB` vào thư mục `data/`.
Lần chạy đầu cần Internet và có thể mất thời gian. Nếu thư viện torchvision trên máy không
tương thích, cài bản PyTorch/torchvision phù hợp với hệ điều hành tại trang chính thức PyTorch.

## 4. Chạy huấn luyện
```powershell
python main.py --download --max-train 0 --max-test 0
```
`0` nghĩa là dùng toàn bộ dữ liệu. Để chạy thử nhanh trước:
```powershell
python main.py --download --max-train 5000 --max-test 2000
```
Kết quả nằm trong `outputs/`; mô hình tốt nhất nằm trong `models/`.

## 5. Chạy giao diện
Sau khi huấn luyện xong:
```powershell
streamlit run app.py
```
Tải lên một ảnh biển báo để xem lớp dự đoán và xác suất (LinearSVC dùng hiệu chỉnh xác suất
CalibratedClassifierCV nếu có; các mô hình khác dùng predict_proba).

## 6. Cấu trúc
- `main.py`: điều phối tải dữ liệu, huấn luyện và đánh giá.
- `src/data.py`: nạp GTSRB và tiền xử lý.
- `src/features.py`: Pixel/HOG.
- `src/models.py`: định nghĩa mô hình.
- `src/evaluate.py`: chỉ số và biểu đồ.
- `app.py`: giao diện dự đoán ảnh mới.

## Lưu ý học thuật
Chỉ ghi vào báo cáo các chỉ số thực tế được sinh ra trong `outputs/metrics.csv`.
Không sử dụng số liệu tham khảo trên Internet như thể là kết quả tự chạy của nhóm.
