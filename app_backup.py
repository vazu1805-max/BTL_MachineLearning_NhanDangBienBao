from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from src.data import preprocess_uploaded_image
from src.features import transform_features

st.set_page_config(page_title="Nhận dạng biển báo giao thông", page_icon="🚦", layout="centered")
st.title("🚦 Nhận dạng biển báo giao thông")
st.write("Tải lên một ảnh biển báo để mô hình Machine Learning dự đoán loại biển báo.")

model_files = sorted(Path("models").glob("*.joblib"))
if not model_files:
    st.warning("Chưa tìm thấy mô hình. Hãy chạy `python main.py --download` trước.")
    st.stop()

chosen = st.selectbox("Chọn mô hình", [p.name for p in model_files])
uploaded = st.file_uploader("Chọn ảnh biển báo", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Ảnh đầu vào", use_container_width=True)
    artifact = joblib.load(Path("models") / chosen)
    x = preprocess_uploaded_image(image)
    x = transform_features(x, artifact["feature"])
    prediction = int(artifact["model"].predict(x)[0])
    st.success(f"Nhãn dự đoán: lớp {prediction}")
    model = artifact["model"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        confidence = float(np.max(probabilities))
        st.metric("Độ tin cậy mô hình", f"{confidence*100:.2f}%")
        st.caption("Độ tin cậy là xác suất do mô hình ước lượng, không phải bảo đảm dự đoán đúng.")
    st.info("GTSRB sử dụng nhãn số từ 0 đến 42. Tra cứu tên biển báo theo bảng lớp GTSRB để diễn giải ý nghĩa.")
