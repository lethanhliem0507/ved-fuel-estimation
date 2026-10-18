import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CẤU HÌNH TRANG & TIÊU ĐỀ
# ==========================================
st.set_page_config(
    page_title="Demo Ước lượng Nhiên liệu VED - Nhóm 5",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
    <div style='background-color: #1a5276; padding: 18px; border-radius: 8px; color: white; text-align: center;'>
        <h2 style='margin: 0; color: white;'>DASHBOARD ƯỚC LƯỢNG MỨC TIÊU THỤ NHIÊN LIỆU THEO CHUYẾN (GBR)</h2>
        <p style='margin: 5px 0 0 0; font-size: 15px;'>
            Đề tài 02: Ước lượng mức tiêu thụ nhiên liệu theo chuyến từ dữ liệu VED | Nhóm 5 - Lớp 23DODA2 (HUTECH)
        </p>
    </div>
""", unsafe_allow_html=True)

# Ghi rõ đây là ước lượng sau chuyến theo yêu cầu bắt buộc của đề tài
st.warning(
    "⚠️ **GHI CHÚ PHẠM VI BÀI TOÁN:** Đây là ứng dụng **Ước lượng sau chuyến đi (Post-trip Estimation)** "
    "từ các đặc trưng vận hành đã ghi nhận qua OBD-II/CAN-Bus. Không sử dụng để dự báo trước chuyến đi (Pre-trip Prediction) "
    "vì tốc độ thực đo và tải động cơ thực tế chỉ thu thập được sau khi chuyến đi kết thúc."
)

# ==========================================
# 2. HÀM TẢI FILE MODEL.PKL
# ==========================================
@st.cache_resource
def load_trained_model(filepath="model.pkl"):
    if not os.path.exists(filepath):
        return None, f"Không tìm thấy file `{filepath}` trong thư mục hiện tại."
    try:
        model = joblib.load(filepath)
        return model, None
    except Exception:
        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            return model, None
        except Exception as e:
            return None, f"Lỗi khi đọc file `{filepath}`: {e}"

model, error_msg = load_trained_model("model.pkl")

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("💡 Hãy đảm bảo bạn đã đặt file `model.pkl` cùng thư mục với file `app.py`, hoặc tải file `model.pkl` lên ở thanh bên trái (Sidebar).")
    uploaded_pkl = st.sidebar.file_uploader("Tải lên file model.pkl", type=["pkl", "joblib"])
    if uploaded_pkl is not None:
        model = joblib.load(uploaded_pkl)
        st.sidebar.success("✅ Đã tải model.pkl thành công!")
    else:
        st.stop()
else:
    st.sidebar.success("✅ Đã kết nối thành công với `model.pkl`")

st.sidebar.markdown("### 📌 Thông tin Mô hình (GBR)")
st.sidebar.write("- **Thuật toán:** Gradient Boosting Regressor")
st.sidebar.write("- **MAE (Test):** `1.894 L/100km`")
st.sidebar.write("- **RMSE (Test):** `2.469 L/100km`")
st.sidebar.write("- **R² (Test):** `0.427`")
st.sidebar.write("- **Dung sai ±1.0 L/100km:** `33.84%`")

# Hàm chuẩn bị DataFrame khớp chính xác với các cột lúc huấn luyện model.pkl
def build_input_dataframe(model_obj, raw_dict, veh_class_label="Car"):
    df_in = pd.DataFrame([raw_dict])
    if hasattr(model_obj, "feature_names_in_"):
        expected_cols = list(model_obj.feature_names_in_)
        for col in expected_cols:
            if col not in df_in.columns:
                if "Class" in col or "Type" in col:
                    df_in[col] = veh_class_label
                else:
                    df_in[col] = 0.0
        return df_in[expected_cols]
    return df_in[['avg_speed_kmh', 'avg_load_pct', 'stop_ratio', 'avg_temp_c', 'total_dist_km', 'total_time_sec']]

# ==========================================
# 3. CHIA TAB CHỨC NĂNG TRÊN GIAO DIỆN
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "🎛️ 1. Ước lượng Chuyến đi Đơn lẻ",
    "📂 2. Ước lượng Lô dữ liệu (CSV) & Bắt lỗi Ngoại lệ",
    "📊 3. Biểu đồ Sai số theo Tải động cơ & Tốc độ"
])

# ------------------------------------------
# TAB 1: NHẬP ĐẶC TRƯNG TỔNG HỢP CỦA CHUYẾN
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.15], gap="large")

    with col1:
        st.subheader("🛠️ Thông số vận hành chuyến đi")
        
        # Các điều khiển tương ứng với ipywidgets trong Project_Group_5.ipynb
        veh_class_full = st.selectbox(
            "Loại xe:",
            ["Car (Xe con)", "SUV (Xe thể thao đa dụng)", "Truck (Xe tải)", "Van/Minivan"],
            index=0
        )
        veh_class_code = veh_class_full.split(" ")[0]

        avg_speed = st.slider("Tốc độ trung bình (km/h):", min_value=0.0, max_value=120.0, value=40.0, step=1.0)
        avg_load = st.slider("Tải động cơ trung bình (%):", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        total_time = st.slider("Thời gian chuyến đi (giây):", min_value=60, max_value=10800, value=1800, step=60)
        stop_ratio = st.slider("Tỷ lệ dừng đỗ (0-1):", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

        is_temp_missing = st.checkbox("Giả lập thiếu cảm biến Nhiệt độ môi trường (NaN)", value=False)
        if is_temp_missing:
            avg_temp = np.nan
            st.caption("ℹ️ Đang truyền `NaN` vào biến `avg_temp_c` để kiểm tra bước `SimpleImputer(strategy='median')` trong Pipeline.")
        else:
            avg_temp = st.slider("Nhiệt độ môi trường (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)

        # Tính quãng đường tích phân từ tốc độ trung bình và thời gian
        total_dist = (avg_speed * total_time) / 3600.0
        st.info(f"📏 **Quãng đường chuyến đi ước tính:** `{total_dist:.2f} km` (Thời gian: `{total_time/60:.1f} phút`)")

    with col2:
        st.subheader("📈 Kết quả Ước lượng Tiêu thụ Nhiên liệu")

        # Kiểm tra ngưỡng lọc chất lượng dữ liệu (Quãng đường >= 1km, Thời gian >= 180s)
        validation_errors = []
        if total_time < 180:
            validation_errors.append(f"Thời gian chuyến đi ({total_time} giây) ngắn hơn ngưỡng tối thiểu 180 giây (3 phút).")
        if total_dist < 1.0:
            validation_errors.append(f"Quãng đường ({total_dist:.2f} km) ngắn hơn ngưỡng tối thiểu 1.0 km (loại bỏ để tránh sai số mẫu số tiệm cận 0).")
        if avg_speed <= 0:
            validation_errors.append("Tốc độ trung bình phải lớn hơn 0 km/h.")

        if validation_errors:
            st.error("❌ **CẢNH BÁO DỮ LIỆU ĐẦU VÀO NGOÀI PHẠM VI HỢP LỆ:**")
            for err in validation_errors:
                st.write(f"- {err}")
        else:
            raw_features = {
                'avg_speed_kmh': avg_speed,
                'avg_load_pct': avg_load,
                'stop_ratio': stop_ratio,
                'avg_temp_c': avg_temp,
                'total_dist_km': total_dist,
                'total_time_sec': float(total_time)
            }
            X_input = build_input_dataframe(model, raw_features, veh_class_code)
            pred_fuel_l100 = float(model.predict(X_input)[0])
            total_fuel_liters = (pred_fuel_l100 * total_dist) / 100.0

            # Đối chiếu với cụm hành trình K-Means trong Báo cáo (Bảng 4.1)
            if avg_speed < 35.0 or stop_ratio >= 0.25:
                cluster_info = "Cụm 0: Đô thị / Kẹt xe (TB cụm: 11.32 L/100km)"
            elif avg_speed < 58.0:
                cluster_info = "Cụm 1: Đường hỗn hợp (TB cụm: 8.78 L/100km)"
            else:
                cluster_info = "Cụm 2: Đường cao tốc (TB cụm: 7.83 L/100km)"

            c1, c2 = st.columns(2)
            c1.metric(
                label="Mức tiêu thụ nhiên liệu (L/100km)",
                value=f"{pred_fuel_l100:.2f} L/100km",
                delta=f"{pred_fuel_l100 - 9.62:+.2f} L so với Baseline (9.62)",
                delta_color="inverse"
            )
            c2.metric(
                label="Tổng lượng xăng tiêu thụ cả chuyến",
                value=f"{total_fuel_liters:.3f} Lít"
            )

            # Đồng hồ trực quan hóa mức tiêu hao
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_fuel_l100,
                title={'text': "Mức tiêu thụ quy đổi (L/100km)"},
                gauge={
                    'axis': {'range': [0, 30]},
                    'bar': {'color': "#1a5276"},
                    'steps': [
                        {'range': [0, 8.5], 'color': "#a9dfbf"},
                        {'range': [8.5, 12.5], 'color': "#f9e79f"},
                        {'range': [12.5, 30], 'color': "#f5b7b1"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9.62
                    }
                }
            ))
            fig_gauge.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.success(f"📌 **Nhận diện trạng thái hành trình:** {cluster_info}")
            st.markdown("""
            **Giải thích kết quả từ mô hình:**
            * Theo phân tích **SHAP Values**, `avg_speed_kmh` (tốc độ trung bình) và `avg_load_pct` (tải động cơ) là hai biến tác động mạnh nhất đến đầu ra.
            * Đường đỏ trên đồng hồ biểu thị mức trung bình Baseline (`9.62 L/100km`).
            """)

# ------------------------------------------
# TAB 2: DỰ ĐOÁN LÔ DỮ LIỆU & KIỂM THỬ NGOẠI LỆ
# ------------------------------------------
with tab2:
    st.subheader("📂 Ước lượng theo Lô (Batch Input) & Kiểm tra Định dạng/Ngoại lệ")
    st.write("Hỗ trợ tải lên file `.csv` chứa nhiều chuyến đi, tự động phát hiện dữ liệu thiếu cảm biến (`NaN`), sai đơn vị hoặc vi phạm ngưỡng độ dài chuyến.")

    sample_df = pd.DataFrame([
        {"Ma_Chuyen": "Trip_01_CaoToc", "avg_speed_kmh": 71.6, "avg_load_pct": 49.0, "stop_ratio": 0.09, "avg_temp_c": 22.0, "total_dist_km": 7.72, "total_time_sec": 382.8},
        {"Ma_Chuyen": "Trip_02_DoThi", "avg_speed_kmh": 31.7, "avg_load_pct": 41.5, "stop_ratio": 0.18, "avg_temp_c": 25.0, "total_dist_km": 2.45, "total_time_sec": 283.9},
        {"Ma_Chuyen": "Trip_03_ThieuNhietDo", "avg_speed_kmh": 34.4, "avg_load_pct": 41.2, "stop_ratio": 0.06, "avg_temp_c": np.nan, "total_dist_km": 3.24, "total_time_sec": 335.1},
        {"Ma_Chuyen": "Trip_04_Loi_QuaNgan", "avg_speed_kmh": 25.0, "avg_load_pct": 30.0, "stop_ratio": 0.30, "avg_temp_c": 24.0, "total_dist_km": 0.50, "total_time_sec": 120.0},
        {"Ma_Chuyen": "Trip_05_Loi_SaiDonViStop", "avg_speed_kmh": 45.0, "avg_load_pct": 32.0, "stop_ratio": 30.0, "avg_temp_c": 26.0, "total_dist_km": 5.00, "total_time_sec": 400.0}
    ])

    st.download_button(
        label="⬇️ Tải file CSV mẫu để thử nghiệm",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="ved_batch_sample.csv",
        mime="text/csv"
    )

    uploaded_csv = st.file_uploader("Chọn file CSV chứa danh sách chuyến đi:", type=["csv"])
    batch_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else sample_df.copy()

    if st.button("▶️ Thực hiện Kiểm tra & Ước lượng Lô dữ liệu", type="primary"):
        req_cols = ['avg_speed_kmh', 'avg_load_pct', 'stop_ratio', 'avg_temp_c', 'total_dist_km', 'total_time_sec']
        missing_c = [c for c in req_cols if c not in batch_df.columns]
        
        if missing_c:
            st.error(f"❌ Định dạng file không hợp lệ! Thiếu các cột: {missing_c}")
        else:
            results = []
            notes = []
            for _, row in batch_df.iterrows():
                if row['total_dist_km'] < 1.0 or row['total_time_sec'] < 180.0:
                    results.append(np.nan)
                    notes.append("❌ Từ chối: Chuyến đi quá ngắn (< 1km hoặc < 180s)")
                elif not (0.0 <= row['stop_ratio'] <= 1.0):
                    results.append(np.nan)
                    notes.append("❌ Từ chối: Sai đơn vị stop_ratio (phải nằm trong [0, 1])")
                elif not (0.0 < row['avg_speed_kmh'] <= 180.0):
                    results.append(np.nan)
                    notes.append("❌ Từ chối: Tốc độ ngoài phạm vi vật lý")
                else:
                    row_input = build_input_dataframe(model, row[req_cols].to_dict())
                    pred_val = float(model.predict(row_input)[0])
                    results.append(round(pred_val, 2))
                    if pd.isna(row['avg_temp_c']):
                        notes.append("⚠️ Hợp lệ (Đã điền khuyết NaN nhiệt độ bằng Median)")
                    else:
                        notes.append("✅ Hợp lệ")

            batch_df["Trạng thái Kiểm định"] = notes
            batch_df["Mức tiêu thụ Ước lượng (L/100km)"] = results
            st.dataframe(batch_df, use_container_width=True)

# ------------------------------------------
# TAB 3: BIỂU ĐỒ SAI SỐ PHÂN TẦNG & HIỆU NĂNG
# ------------------------------------------
with tab3:
    st.subheader("📊 Phân tích Sai số theo Tải động cơ và Tốc độ (Tập Test độc lập: 3.026 chuyến)")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Bảng 5.2: Biểu đồ nhiệt (Heatmap) Sai số MAE theo Tốc độ & Tải động cơ**")
        mae_matrix = [
            [2.73, 2.49, np.nan],
            [1.67, 1.83, 2.16],
            [1.71, 1.32, 0.98]
        ]
        fig_hm = px.imshow(
            mae_matrix,
            x=["< 30% (Tải nhẹ)", "30 - 50% (Tải vừa)", "> 50% (Tải nặng)"],
            y=["0 - 30 km/h", "30 - 60 km/h", "> 60 km/h"],
            text_auto=True,
            color_continuous_scale="OrRd",
            labels=dict(x="Nhóm Tải động cơ (avg_load_pct)", y="Nhóm Tốc độ (avg_speed_kmh)", color="MAE (L/100km)")
        )
        fig_hm.update_layout(height=360)
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_b:
        st.markdown("**Bảng 5.1: So sánh MAE và RMSE giữa các mô hình trên tập Test**")
        perf_df = pd.DataFrame({
            "Mô hình": ["Gradient Boosting (GBR)", "Random Forest", "Ridge", "Linear Regression", "Baseline (Mean)"],
            "MAE": [1.894, 1.945, 2.191, 2.192, 2.515],
            "RMSE": [2.469, 2.545, 2.966, 2.966, 3.287]
        })
        fig_comp = px.bar(
            perf_df, x="Mô hình", y=["MAE", "RMSE"],
            barmode="group", text_auto=True,
            color_discrete_sequence=["#27ae60", "#c0392b"]
        )
        fig_comp.update_layout(height=360)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.info(
        "💡 **Nhận xét chuyên môn (Chương 5 Báo cáo):**\n"
        "- **Vùng sai số cao nhất:** Dải tốc độ chậm `0 - 30 km/h` có MAE cao nhất (`2.49 - 2.73 L/100km`) do phương tiện di chuyển trong đô thị kẹt xe, dừng/khởi hành liên tục.\n"
        "- **Nguyên nhân ô giá trị `NaN` tại (Tốc độ `0 - 30 km/h`, Tải `> 50%`):** Ở tốc độ thấp, động cơ hoạt động ở vòng tua thấp và không chịu lực cản gió lớn nên tải động cơ trung bình cả chuyến chỉ dao động từ 15%–35%. Trong 3.026 chuyến đi của tập Test, không có mẫu nào rơi vào tổ hợp này (N = 0)."
    )
