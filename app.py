from datetime import datetime
import hashlib
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image

from src.data import preprocess_uploaded_image
from src.features import transform_features


# ==============================
# 1. CAU HINH GIAO DIEN
# ==============================

st.set_page_config(
    page_title="He thong nhan dang bien bao giao thong",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG")
st.caption(
    "Đề tài Machine Learning - Nhận dạng và giải thích ý nghĩa biển báo giao thông"
)

st.markdown("---")


# ==============================
# 2. BANG THONG TIN 43 LOP GTSRB
# ==============================

SIGN_INFO = {
    0: ("Giới hạn tốc độ 20 km/h", "Speed limit (20km/h)",
        "Phương tiện không được chạy vượt quá 20 km/h."),
    1: ("Giới hạn tốc độ 30 km/h", "Speed limit (30km/h)",
        "Phương tiện không được chạy vượt quá 30 km/h."),
    2: ("Giới hạn tốc độ 50 km/h", "Speed limit (50km/h)",
        "Phương tiện không được chạy vượt quá 50 km/h."),
    3: ("Giới hạn tốc độ 60 km/h", "Speed limit (60km/h)",
        "Phương tiện không được chạy vượt quá 60 km/h."),
    4: ("Giới hạn tốc độ 70 km/h", "Speed limit (70km/h)",
        "Phương tiện không được chạy vượt quá 70 km/h."),
    5: ("Giới hạn tốc độ 80 km/h", "Speed limit (80km/h)",
        "Phương tiện không được chạy vượt quá 80 km/h."),
    6: ("Hết giới hạn tốc độ 80 km/h", "End of speed limit (80km/h)",
        "Báo hiệu kết thúc giới hạn tốc độ 80 km/h."),
    7: ("Giới hạn tốc độ 100 km/h", "Speed limit (100km/h)",
        "Phương tiện không được chạy vượt quá 100 km/h."),
    8: ("Giới hạn tốc độ 120 km/h", "Speed limit (120km/h)",
        "Phương tiện không được chạy vượt quá 120 km/h."),
    9: ("Cấm vượt", "No passing",
        "Cấm các phương tiện thực hiện hành vi vượt xe theo quy định."),
    10: ("Cấm xe tải vượt", "No passing for vehicles over 3.5 tons",
         "Cấm xe tải có khối lượng quy định trên 3,5 tấn vượt xe khác."),
    11: ("Giao nhau với đường ưu tiên", "Right-of-way at the next intersection",
         "Báo hiệu sắp đến nơi giao nhau với đường ưu tiên."),
    12: ("Đường ưu tiên", "Priority road",
         "Báo hiệu đường đang đi là đường ưu tiên."),
    13: ("Nhường đường", "Yield",
         "Người điều khiển phương tiện phải nhường đường cho xe trên đường ưu tiên."),
    14: ("Dừng lại", "Stop",
         "Người điều khiển phương tiện phải dừng lại và quan sát trước khi đi tiếp."),
    15: ("Cấm các loại xe", "No vehicles",
         "Cấm các loại phương tiện giao thông đi vào, trừ trường hợp được phép."),
    16: ("Cấm xe tải trên 3,5 tấn", "Vehicles over 3.5 tons prohibited",
         "Cấm xe tải có khối lượng quy định trên 3,5 tấn đi vào."),
    17: ("Cấm đi vào", "No Entry",
         "Cấm các phương tiện đi vào đoạn đường có đặt biển báo này."),
    18: ("Đường có nguy hiểm", "General caution",
         "Cảnh báo đoạn đường phía trước có nguy hiểm cần chú ý."),
    19: ("Đường cong nguy hiểm bên trái", "Dangerous curve to the left",
         "Cảnh báo đoạn đường phía trước cong nguy hiểm về bên trái."),
    20: ("Đường cong nguy hiểm bên phải", "Dangerous curve to the right",
         "Cảnh báo đoạn đường phía trước cong nguy hiểm về bên phải."),
    21: ("Đường cong kép", "Double curve",
         "Cảnh báo đoạn đường có hai khúc cua liên tiếp."),
    22: ("Đường gồ ghề", "Bumpy road",
         "Cảnh báo mặt đường không bằng phẳng, cần giảm tốc độ."),
    23: ("Đường trơn trượt", "Slippery road",
         "Cảnh báo mặt đường có thể trơn trượt, cần điều khiển xe cẩn thận."),
    24: ("Đường bị thu hẹp bên phải", "Road narrows on the right",
         "Cảnh báo phần đường phía trước bị thu hẹp ở bên phải."),
    25: ("Công trường đang thi công", "Road work",
         "Cảnh báo khu vực phía trước đang thi công, cần chú ý và giảm tốc độ."),
    26: ("Có tín hiệu đèn giao thông", "Traffic signals",
         "Cảnh báo phía trước có hệ thống đèn tín hiệu giao thông."),
    27: ("Người đi bộ", "Pedestrians",
         "Cảnh báo khu vực có người đi bộ qua đường."),
    28: ("Trẻ em qua đường", "Children crossing",
         "Cảnh báo khu vực thường có trẻ em qua đường."),
    29: ("Đường có người đi xe đạp cắt ngang", "Bicycles crossing",
         "Cảnh báo phía trước có người đi xe đạp cắt ngang."),
    30: ("Cảnh báo băng tuyết", "Beware of ice or snow",
         "Cảnh báo mặt đường có thể có băng hoặc tuyết gây trơn trượt."),
    31: ("Động vật hoang dã qua đường", "Wild animals crossing",
         "Cảnh báo có thể có động vật hoang dã băng qua đường."),
    32: ("Hết mọi lệnh cấm và giới hạn tốc độ", "End of all speed and passing limits",
         "Báo hiệu kết thúc các lệnh cấm vượt và giới hạn tốc độ tương ứng."),
    33: ("Chỉ được rẽ phải", "Turn right ahead",
         "Báo hiệu hướng đi bắt buộc là rẽ phải."),
    34: ("Chỉ được rẽ trái", "Turn left ahead",
         "Báo hiệu hướng đi bắt buộc là rẽ trái."),
    35: ("Đi thẳng", "Ahead only",
         "Báo hiệu phương tiện chỉ được đi thẳng."),
    36: ("Đi thẳng hoặc rẽ phải", "Go straight or right",
         "Báo hiệu phương tiện được đi thẳng hoặc rẽ phải."),
    37: ("Đi thẳng hoặc rẽ trái", "Go straight or left",
         "Báo hiệu phương tiện được đi thẳng hoặc rẽ trái."),
    38: ("Đi bên phải", "Keep right",
         "Yêu cầu phương tiện đi về phía bên phải theo hướng mũi tên."),
    39: ("Đi bên trái", "Keep left",
         "Yêu cầu phương tiện đi về phía bên trái theo hướng mũi tên."),
    40: ("Vòng xuyến bắt buộc", "Roundabout mandatory",
         "Báo hiệu phương tiện phải đi theo hướng vòng xuyến."),
    41: ("Hết cấm vượt", "End of no passing",
         "Báo hiệu kết thúc quy định cấm vượt tương ứng."),
    42: ("Hết cấm xe tải vượt", "End of no passing by vehicles over 3.5 tons",
         "Báo hiệu kết thúc quy định cấm xe tải trên 3,5 tấn vượt xe khác.")
}


# ==============================
# 3. KIEM TRA MO HINH
# ==============================

model_files = sorted(Path("models").glob("*.joblib"))

if not model_files:
    st.warning("Chưa tìm thấy mô hình. Hãy chạy python main.py trước.")
    st.stop()


# ==============================
# 4. LUA CHON MO HINH
# ==============================

st.subheader("⚙️ Cấu hình nhận dạng")

chosen = st.selectbox(
    "Chọn mô hình Machine Learning:",
    [p.name for p in model_files],
    index=(
        [p.name for p in model_files].index("LinearSVC_hog.joblib")
        if "LinearSVC_hog.joblib" in [p.name for p in model_files]
        else 0
    )
)

st.info(
    "Mô hình LinearSVC kết hợp đặc trưng HOG là mô hình đạt kết quả tốt nhất "
    "trong thí nghiệm hiện tại của dự án."
)


# ==============================
# 5. TAI ANH BIEN BAO
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

if "processed_images" not in st.session_state:
    st.session_state.processed_images = set()

st.subheader("📷 Tải ảnh biển báo")

uploaded = st.file_uploader(
    "Chọn ảnh biển báo giao thông để hệ thống nhận dạng",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.markdown("---")
    st.subheader("🔍 Kết quả phân tích")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(
            image,
            caption="Ảnh biển báo đầu vào",
            use_container_width=True
        )

    with col2:

        with st.spinner("Hệ thống đang phân tích ảnh..."):

            artifact = joblib.load(Path("models") / chosen)

            x = preprocess_uploaded_image(image)
            x = transform_features(x, artifact["feature"])

            prediction = int(artifact["model"].predict(x)[0])
            image_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

    if image_hash not in st.session_state.processed_images:
        st.session_state.processed_images.add(image_hash)

        st.session_state.history.insert(0, {
            "Thời gian": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            "Tên ảnh": uploaded.name,
            "Mã lớp": prediction,
            "Biển báo": SIGN_INFO[prediction][0]
        })

        info = SIGN_INFO.get(prediction)

        if info is None:
            st.error("Không tìm thấy thông tin của lớp dự đoán.")
            st.stop()

        vietnamese_name, english_name, meaning = info

        st.success("NHẬN DẠNG THÀNH CÔNG")

        st.markdown(f"## 🚦 {vietnamese_name.upper()}")

        st.markdown(f"**Mã lớp GTSRB:** {prediction}")
        st.markdown(f"**Tên tiếng Anh:** {english_name}")

        st.markdown("### 📖 Ý nghĩa biển báo")
        st.write(meaning)

        model = artifact["model"]

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(x)[0]
            confidence = float(np.max(probabilities))

            st.markdown("### 📊 Độ tin cậy mô hình")
            st.progress(confidence)
            st.metric(
                label="Xác suất dự đoán cao nhất",
                value=f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Đây là xác suất do mô hình ước lượng, không phải bảo đảm "
                "rằng dự đoán chắc chắn đúng."
            )

        else:
            st.info("Mô hình này không cung cấp xác suất dự đoán.")

    st.markdown("---")

    st.caption(
        "Lưu ý: Hệ thống sử dụng bộ dữ liệu GTSRB gồm 43 lớp biển báo của Đức. "
        "Tên và ý nghĩa được diễn giải theo nhãn của bộ dữ liệu, "
        "có thể khác với cách phân loại biển báo tại Việt Nam."
    )

else:
    st.info(
        "👆 Vui lòng tải lên một ảnh biển báo để bắt đầu quá trình nhận dạng."
    )


# ==============================
# 6. THONG TIN DO AN
# ==============================

st.markdown("---")
st.subheader("📚 Thông tin đề tài")

st.write("""
- **Tên đề tài:** Nhận dạng biển báo giao thông.
- **Lĩnh vực:** Machine Learning.
- **Bộ dữ liệu:** German Traffic Sign Recognition Benchmark (GTSRB).
- **Số lượng lớp:** 43 lớp biển báo.
- **Các thuật toán:** KNN, Decision Tree, LinearSVC.
- **Đặc trưng:** Pixel và HOG.
""")
st.markdown("---")
st.header("📜 LỊCH SỬ NHẬN DẠNG")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.history = []
        st.session_state.processed_images = set()
        st.rerun()
else:
    st.info("Chưa có lịch sử nhận dạng.")
