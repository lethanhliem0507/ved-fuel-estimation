import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Ước Lượng Nhiên Liệu VED - Nhóm 5",
    page_icon="🚗",
    layout="wide"
)

# 2. TẢI MÔ HÌNH VỚI CƠ CHẾ CACHE
@st.cache_resource
def load_model():
    return joblib.load('gbr_pipeline.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Không thể tải file mô hình 'gbr_pipeline.joblib'. Lỗi: {e}")
    st.stop()

# 3. QUY TẮC HỢP ĐỒNG DỮ LIỆU (DATA CONTRACT)
VALIDATION_RULES = {
    'avg_speed_kmh': {'min': 0.0, 'max': 150.0, 'unit': 'km/h', 'name': 'Tốc độ trung bình'},
    'avg_load_pct':  {'min': 0.0, 'max': 100.0, 'unit': '%',    'name': 'Tải động cơ trung bình'},
    'stop_ratio':    {'min': 0.0, 'max': 1.0,   'unit': 'tỷ lệ','name': 'Tỷ lệ thời gian dừng'},
    'avg_temp_c':    {'min': -30.0, 'max': 55.0,'unit': '°C',   'name': 'Nhiệt độ môi trường'},
    'total_dist_km': {'min': 0.1, 'max': 500.0, 'unit': 'km',   'name': 'Tổng quãng đường'},
    'total_time_sec':{'min': 60.0, 'max': 86400.0,'unit':'giây', 'name': 'Tổng thời gian đi'}
}

def validate_and_clean(df):
    clean_df = df.copy()
    warnings = []
    for col, rule in VALIDATION_RULES.items():
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            if clean_df[col].isnull().any():
                warnings.append(f"⚠️ Cột **{rule['name']}** có giá trị khuyết (đã điền bằng trung vị).")
            # Clip dải hợp lệ
            out_bounds = (clean_df[col] < rule['min']) | (clean_df[col] > rule['max'])
            if out_bounds.any():
                warnings.append(f"🚨 Cột **{rule['name']}** có giá trị ngoài dải [{rule['min']}-{rule['max']} {rule['unit']}] (đã khống chế dải an toàn).")
                clean_df[col] = clean_df[col].clip(lower=rule['min'], upper=rule['max'])
    return clean_df, warnings

# 4. GIAO DIỆN CHÍNH
st.title("🚗 ƯỚC LƯỢNG MỨC TIÊU THỤ NHIÊN LIỆU SAU CHUYẾN ĐI (POST-TRIP)")
st.caption("Dự án Khai phá Dữ liệu Telematics VED | Nhóm 05 - Lớp 23DODA2")

tab1, tab2 = st.tabs(["📌 Nhập Đơn Chuyến", "📁 Xử Lý Lô CSV"])

# --- TAB 1: NHẬP ĐƠN CHUYẾN ---
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        speed = st.number_input("Tốc độ trung bình (km/h):", min_value=0.0, max_value=150.0, value=38.5, step=0.5)
        load = st.number_input("Tải động cơ trung bình (%):", min_value=0.0, max_value=100.0, value=32.0, step=1.0)
        
    with col2:
        stop_ratio = st.slider("Tỷ lệ thời gian dừng đỗ (0-1):", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
        temp = st.number_input("Nhiệt độ môi trường (°C):", min_value=-20.0, max_value=50.0, value=22.0, step=1.0)
        
    with col3:
        dist = st.number_input("Tổng quãng đường (km):", min_value=0.1, max_value=500.0, value=6.5, step=0.5)
        time_sec = st.number_input("Tổng thời gian (giây):", min_value=60.0, max_value=86400.0, value=600.0, step=10.0)

    if st.button("🚀 Ước Lượng Nhiên Liệu", type="primary"):
        input_data = pd.DataFrame([{
            'avg_speed_kmh': speed,
            'avg_load_pct': load,
            'stop_ratio': stop_ratio,
            'avg_temp_c': temp,
            'total_dist_km': dist,
            'total_time_sec': time_sec
        }])
        
        clean_data, warnings = validate_and_clean(input_data)
        features = ['avg_speed_kmh', 'avg_load_pct', 'stop_ratio', 'avg_temp_c', 'total_dist_km', 'total_time_sec']
        
        pred = model.predict(clean_data[features])[0]
        mae = 1.894 # MAE của GBR trên tập Test

        st.markdown("---")
        st.subheader("📌 Báo Cáo Kết Quả Ước Lượng")
        st.metric(label="Mức tiêu thụ nhiên liệu ước tính", value=f"{pred:.2f} L/100km")
        st.info(f"Khoảng tin cậy tin cậy (±MAE): **{max(2.0, pred - mae):.2f} – {pred + mae:.2f} L/100km**")

        if pred > 11.5:
            st.error("⚠️ **CẢNH BÁO MỨC TÍNH CAO:** Xe vận hành trong điều kiện kẹt xe nặng hoặc ép tải lớn khiến tiêu hao xăng tăng vọt.")
        else:
            st.success("✅ **ĐÁNH GIÁ MỨC TIẾT KIỆM / BÌNH THƯỜNG:** Tốc độ vận hành ổn định và tải động cơ ở mức vừa phải.")

        if warnings:
            with st.expander("Nhật ký chuẩn hóa dữ liệu đầu vào"):
                for w in warnings:
                    st.write(w)

# --- TAB 2: TẢI FILE BATCH CSV ---
with tab2:
    uploaded_file = st.file_uploader("Tải lên file CSV chứa danh sách chuyến đi:", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        clean_batch, warnings = validate_and_clean(df_batch)
        features = ['avg_speed_kmh', 'avg_load_pct', 'stop_ratio', 'avg_temp_c', 'total_dist_km', 'total_time_sec']
        
        preds = model.predict(clean_batch[features])
        clean_batch['Predicted_Fuel_L100'] = np.round(preds, 2)
        
        st.success(f"✅ Đã xử lý {len(clean_batch)} chuyến đi thành công!")
        st.dataframe(clean_batch[['avg_speed_kmh', 'avg_load_pct', 'stop_ratio', 'total_dist_km', 'Predicted_Fuel_L100']])
