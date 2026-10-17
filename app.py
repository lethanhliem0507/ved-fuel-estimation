import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Ước lượng Nhiên liệu VED - Nhóm 05",
    page_icon="🚗",
    layout="wide"
)

# 2. HÀM TẢI MÔ HÌNH PIPELINE
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 3. SIDEBAR: THÔNG TIN ĐỀ TÀI & NHÓM THỰC HIỆN
with st.sidebar:
    st.markdown("## 🏫 TRƯỜNG KỸ THUẬT HUTECH")
    st.markdown("**Môn học:** Khai phá dữ liệu trong AI và Ô tô (EAT114)")
    st.markdown("**Ngành:** Công nghệ Ô tô điện")
    st.markdown("---")
    st.markdown("### 👨‍💻 Nhóm thực hiện (Nhóm 05 - 23DODA2):")
    st.markdown("- **Lê Thành Liêm** (`2389400120`)\n- **Nguyễn Thành Đạt** (`2389400042`)\n- **Nguyễn Hoài Phát** (`2389400153`)")
    st.markdown("**GVHD:** TS. Hoàng Văn Quý")
    st.markdown("---")
    st.info(
        "📌 **Lưu ý kỹ thuật:** Đây là bài toán **Ước lượng sau chuyến đi (Post-trip Estimation)** "
        "từ các đặc trưng vận hành thực đo đã ghi nhận qua OBD-II/CAN-Bus, **không phải** dự báo trước chuyến đi (Pre-trip Prediction)."
    )

# 4. TIÊU ĐỀ CHÍNH
st.title("🚗 HỆ THỐNG ƯỚC LƯỢNG MỨC TIÊU THỤ NHIÊN LIỆU THEO CHUYẾN TỪ DỮ LIỆU VED")
st.warning(
    "⚠️ **Xác định bài toán (Post-trip Estimation):** Hệ thống sử dụng các thông số vận hành tổng hợp "
    "(tốc độ thực đo, tải động cơ thực tế, tỷ lệ dừng đỗ...) thu thập **sau khi chuyến đi đã kết thúc** để ước lượng "
    "mức tiêu hao nhiên liệu quy đổi (L/100km). Tuyệt đối không sử dụng biến `Fuel Rate` ở đầu vào để chống rò rỉ dữ liệu (Data Leakage)."
)

# 5. CHIA TAB NỘI DUNG
tab1, tab2, tab3 = st.tabs([
    "🎛️ Demo Ước lượng Nhiên liệu (GBR)",
    "📊 Phân tích Sai số & Hiệu năng Mô hình",
    "📖 Tổng quan Dữ liệu & Phương pháp"
])

# ==========================================
# TAB 1: DEMO ƯỚC LƯỢNG NHIÊN LIỆU
# ==========================================
with tab1:
    st.subheader("Nhập đặc trưng tổng hợp sau chuyến đi")
    
    col_input, col_result = st.columns([1.2, 1])
    
    with col_input:
        vehicle_class = st.selectbox(
            "Phân khúc xe (Vehicle Class - Tham khảo đối chiếu):",
            ["Car (Xe con)", "SUV (Xe thể thao đa dụng)", "Truck (Xe tải)", "Van/Minivan"]
        )
        
        c1, c2 = st.columns(2)
        with c1:
            avg_speed_kmh = st.slider("Tốc độ trung bình (km/h):", min_value=5.0, max_value=120.0, value=40.0, step=1.0)
            avg_load_pct = st.slider("Tải động cơ trung bình (%):", min_value=5.0, max_value=100.0, value=30.0, step=1.0,
                                     help="Tải động cơ thực tế từ CAN-Bus (Absolute Load %), không phải khối lượng hàng hóa.")
            stop_ratio = st.slider("Tỷ lệ dừng đỗ (0.0 - 0.9):", min_value=0.0, max_value=0.9, value=0.18, step=0.01)
            
        with c2:
            total_time_sec = st.slider("Thời gian chuyến đi (giây):", min_value=180, max_value=7500, value=1800, step=60,
                                       help="Chỉ xét các chuyến đi hợp lệ từ 180 giây (3 phút) trở lên.")
            avg_temp_c = st.slider("Nhiệt độ môi trường (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)
            
            # Tự động ước tính quãng đường từ tốc độ trung bình và thời gian hoặc cho phép hiệu chỉnh
            calc_dist = round((avg_speed_kmh * total_time_sec) / 3600.0, 2)
            total_dist_km = st.number_input("Tổng quãng đường (km):", min_value=1.0, max_value=250.0, value=max(1.0, calc_dist), step=0.5,
                                            help="Mặc định tính theo công thức: V_tb * Thời_gian / 3600 (Tối thiểu >= 1 km).")

    with col_result:
        st.markdown("### 🎯 Kết quả Ước lượng Sau Chuyến")
        
        # Đóng gói đúng 6 đặc trưng theo thứ tự của Pipeline trong model.pkl
        input_df = pd.DataFrame([{
            'avg_speed_kmh': avg_speed_kmh,
            'avg_load_pct': avg_load_pct,
            'stop_ratio': stop_ratio,
            'avg_temp_c': avg_temp_c,
            'total_dist_km': total_dist_km,
            'total_time_sec': float(total_time_sec)
        }])
        
        if model is not None:
            pred_fuel_l100 = float(model.predict(input_df)[0])
        else:
            # Công thức xấp xỉ dự phòng nếu chưa copy file model.pkl vào thư mục
            pred_fuel_l100 = 9.575 - 0.11 * (avg_speed_kmh - 40.9) + 0.08 * (avg_load_pct - 30.8) + 6.5 * (stop_ratio - 0.186)
            pred_fuel_l100 = float(np.clip(pred_fuel_l100, 3.5, 35.0))
            st.caption("*(Đang chạy chế độ mô phỏng do chưa tìm thấy file `model.pkl` trong thư mục)*")

        total_fuel_liters = (pred_fuel_l100 * total_dist_km) / 100.0
        
        # Phân loại trạng thái hành trình dựa trên kết quả K-Means (K=3) trong báo cáo
        if avg_speed_kmh < 35 or stop_ratio >= 0.22:
            cluster_name = "Cụm 0: Đô thị / Kẹt xe (TB chuẩn: 11.32 L/100km)"
        elif avg_speed_kmh < 58:
            cluster_name = "Cụm 1: Đường hỗn hợp (TB chuẩn: 8.78 L/100km)"
        else:
            cluster_name = "Cụm 2: Cao tốc (TB chuẩn: 7.83 L/100km)"

        m1, m2 = st.columns(2)
        m1.metric("Mức tiêu thụ quy đổi", f"{pred_fuel_l100:.2f} L/100km", delta=f"{pred_fuel_l100 - 9.58:.2f} so với TB tập dữ liệu", delta_color="inverse")
        m2.metric("Tổng lượng xăng tiêu thụ", f"{total_fuel_liters:.2f} Lít", f"Quãng đường: {total_dist_km:.2f} km")
        
        st.success(f"🛣️ **Nhận diện trạng thái hành trình (K-Means):** {cluster_name}")
        
        # Biểu đồ đồng hồ đo (Gauge Chart)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_fuel_l100,
            title={'text': "Lít / 100 km (Mô hình GBR)"},
            gauge={
                'axis': {'range': [0, 25]},
                'bar': {'color': "#1a5276"},
                'steps': [
                    {'range': [0, 8.5], 'color': "#abebc6"},
                    {'range': [8.5, 12.0], 'color': "#f9e79f"},
                    {'range': [12.0, 25.0], 'color': "#f5b7b1"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 9.575}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 2: PHÂN TÍCH SAI SỐ & HIỆU NĂNG MÔ HÌNH
# ==========================================
with tab2:
    st.subheader("1. So sánh hiệu năng các mô hình trên tập Test độc lập theo xe (3.026 chuyến - 68 xe)")
    
    df_models = pd.DataFrame({
        "Mô hình": ["Gradient Boosting (GBR)", "Random Forest", "Ridge Regression", "Linear Regression", "Baseline (Mean)"],
        "MAE (L/100km)": [1.894, 1.945, 2.191, 2.192, 2.515],
        "RMSE (L/100km)": [2.469, 2.545, 2.966, 2.966, 3.287],
        "R² Score": [0.427, 0.391, 0.173, 0.173, -0.016],
        "Trong dung sai ±1L (%)": [33.84, 33.77, 28.68, 28.68, 23.83]
    })
    
    col_t1, col_t2 = st.columns([1.1, 1.2])
    with col_t1:
        st.dataframe(df_models, use_container_width=True, hide_index=True)
        st.caption("Mô hình Gradient Boosting Regressor (GBR) đạt hiệu năng cao nhất trên mọi chỉ số đánh giá.")
        
    with col_t2:
        fig_bar = px.bar(
            df_models.melt(id_vars="Mô hình", value_vars=["MAE (L/100km)", "RMSE (L/100km)"]),
            x="Mô hình", y="value", color="variable", barmode="group",
            title="So sánh Sai số MAE và RMSE giữa các Mô hình Hồi quy",
            labels={"value": "Sai số (L/100km)", "variable": "Chỉ số"}
        )
        fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Phân tích sai số MAE phân tầng theo Tốc độ và Tải động cơ (Error Analysis)")
    
    col_e1, col_e2 = st.columns([1, 1.2])
    with col_e1:
        error_matrix = pd.DataFrame({
            "Nhóm Tốc độ \\ Nhóm Tải": ["0 - 30 km/h (Chậm)", "30 - 60 km/h (Trung bình)", "> 60 km/h (Nhanh)"],
            "Tải nhẹ (<30%)": [2.73, 1.67, 1.71],
            "Tải vừa (30-50%)": [2.49, 1.83, 1.32],
            "Tải nặng (>50%)": [np.nan, 2.16, 0.98]
        })
        st.dataframe(error_matrix, use_container_width=True, hide_index=True)
        st.info(
            "💡 **Giải thích giá trị `NaN` ở ô (Tốc độ 0-30 km/h, Tải >50%):** "
            "Ở dải tốc độ chậm đô thị, động cơ hoạt động ở vòng tua thấp, không chịu lực cản gió lớn nên tải trung bình cả chuyến "
            "chỉ dao động từ 15% - 35%. Không có mẫu nào (N = 0) trong tập Test rơi vào ô này."
        )

    with col_e2:
        z_data = [[2.73, 2.49, np.nan], [1.67, 1.83, 2.16], [1.71, 1.32, 0.98]]
        fig_heat = px.imshow(
            z_data,
            x=["Tải nhẹ (<30%)", "Tải vừa (30-50%)", "Tải nặng (>50%)"],
            y=["0-30 km/h", "30-60 km/h", ">60 km/h"],
            text_auto=True, color_continuous_scale="YlOrRd",
            title="Biểu đồ Nhiệt (Heatmap) Sai số MAE (L/100km) theo Tốc độ và Tải động cơ"
        )
        fig_heat.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: TỔNG QUAN DỮ LIỆU & PHƯƠNG PHÁP
# ==========================================
with tab3:
    st.subheader("Quy trình Khai phá Dữ liệu VED (Vehicle Energy Dataset)")
    st.markdown("""
    1. **Lọc và Ghép nối dữ liệu:** Từ 383 xe ban đầu của bộ dữ liệu VED, lọc ra **341 xe chạy xăng (Gasoline)** với **17.982.360 bản ghi** chuỗi thời gian.
    2. **Tích phân số học tính nhãn:** Tích phân `Fuel Rate (L/h)` và `Vehicle Speed (km/h)` theo từng bước thời gian rời rạc `dt_sec` để tính tổng lượng xăng tiêu thụ (Lít), tổng quãng đường (km) và quy đổi ra nhãn mục tiêu `fuel_L100` (L/100km).
    3. **Kiểm soát chất lượng (Data Cleaning):** Từ 27.553 chuyến đi ban đầu, loại bỏ các chuyến quá ngắn (`< 1 km` hoặc `< 180 giây`), đứt quãng tín hiệu (`> 10 giây`), và nhãn dị biệt phi vật lý (`< 2.0` hoặc `> 40.0 L/100km`). Thu được **12.199 chuyến đi hợp lệ**.
    4. **Phân tách chống rò rỉ dữ liệu (Group Split):** Sử dụng `GroupShuffleSplit` theo khóa `VehId` với tỷ lệ 75% Train (**9.173 chuyến / 201 xe**) và 25% Test (**3.026 chuyến / 68 xe**), đảm bảo đánh giá mô hình trên các xe hoàn toàn mới chưa từng xuất hiện trong tập huấn luyện.
    """)
    
    st.subheader("Thống kê mô tả 12.199 chuyến đi hợp lệ")
    stats_df = pd.DataFrame({
        "Đặc trưng": ["fuel_L100 (L/100km)", "avg_speed_kmh (km/h)", "avg_load_pct (%)", "stop_ratio", "total_dist_km (km)", "total_time_sec (s)"],
        "Mean": [9.575, 40.901, 30.865, 0.186, 5.627, 552.671],
        "Median": [9.037, 38.900, 30.404, 0.177, 4.755, 453.200],
        "Std": [3.424, 14.362, 7.378, 0.134, 3.677, 375.070],
        "Min": [2.113, 7.660, 8.576, 0.000, 1.004, 180.000],
        "Q1 (25%)": [7.372, 31.870, 27.142, 0.078, 3.115, 301.850],
        "Q3 (75%)": [11.274, 46.521, 33.921, 0.268, 6.975, 682.650],
        "Max": [39.868, 115.030, 387.926, 0.846, 64.207, 7408.400]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
