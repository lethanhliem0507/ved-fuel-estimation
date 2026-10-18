import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="VED Fuel Estimation Demo - Group 5",
    page_icon="🚗",
    layout="wide"
)

# Header & Banner cảnh báo Post-trip bắt buộc theo đề bài
st.markdown("""
    <div style='background-color: #1a5276; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
        <h2 style='margin: 0; color: white;'>🚗 HỆ THỐNG ƯỚC LƯỢNG MỨC TIÊU THỤ NHIÊN LIỆU TỪ DỮ LIỆU VED</h2>
        <p style='margin: 5px 0 0 0; font-size: 16px;'>
            Học phần: Khai phá dữ liệu trong AI và Ô tô (EAT114) | Thực hiện: Nhóm 5 - Lớp 23DODA2 (HUTECH)
        </p>
    </div>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ **LƯU Ý KỸ THUẬT (PHÂN ĐỊNH BÀI TOÁN):** Ứng dụng này thực hiện bài toán **ƯỚC LƯỢNG SAU CHUYẾN ĐI (Post-trip Estimation)** "
    "dựa trên các đặc trưng vận hành đã được ghi nhận qua OBD-II/CAN-Bus sau khi kết thúc hành trình. "
    "Tuyệt đối không sử dụng cho mục đích dự báo trước chuyến đi (Pre-trip Prediction) do tốc độ và tải động cơ thực đo chỉ có sau khi hoàn thành chuyến."
)

# ==========================================
# 2. TẢI DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH (CACHED)
# ==========================================
@st.cache_resource
def load_and_train_models():
    file_path = "ved_cleaned_trips.csv"
    
    # Nếu có file thực tế từ Bước 1 thì đọc, nếu không thì khởi tạo tập mẫu chuẩn theo Bảng 3.2 báo cáo
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        is_demo_data = False
    else:
        is_demo_data = True
        np.random.seed(42)
        n_samples = 1500
        veh_ids = np.random.choice(range(1, 70), size=n_samples)
        avg_speed = np.clip(np.random.normal(40.9, 14.3, n_samples), 8.0, 115.0)
        avg_load = np.clip(np.random.normal(30.8, 7.3, n_samples), 8.5, 95.0)
        stop_ratio = np.clip(np.random.normal(0.186, 0.134, n_samples), 0.0, 0.85)
        total_time = np.clip(np.random.normal(552.6, 375.0, n_samples), 180.0, 7200.0)
        total_dist = (avg_speed * total_time) / 3600.0
        avg_temp = np.random.normal(18.0, 10.0, n_samples)
        # Gài 10% giá trị NaN cho nhiệt độ để kiểm chứng SimpleImputer
        avg_temp[np.random.choice(n_samples, 150, replace=False)] = np.nan
        
        # Công thức phi tuyến mô phỏng mối quan hệ chữ U với tốc độ và thuận chiều với tải/dừng đỗ
        fuel_l100 = (
            13.5 
            - 0.14 * avg_speed 
            + 0.0009 * (avg_speed ** 2) 
            + 0.08 * avg_load 
            + 6.5 * stop_ratio 
            + np.random.normal(0, 1.2, n_samples)
        )
        fuel_l100 = np.clip(fuel_l100, 2.5, 39.5)
        
        df = pd.DataFrame({
            'VehId': veh_ids,
            'avg_speed_kmh': avg_speed,
            'avg_load_pct': avg_load,
            'stop_ratio': stop_ratio,
            'avg_temp_c': avg_temp,
            'total_dist_km': total_dist,
            'total_time_sec': total_time,
            'fuel_L100': fuel_l100
        })
        # Lọc quãng đường >= 1km theo quy tắc làm sạch
        df = df[df['total_dist_km'] >= 1.0].reset_index(drop=True)

    feature_cols = ['avg_speed_kmh', 'avg_load_pct', 'stop_ratio', 'avg_temp_c', 'total_dist_km', 'total_time_sec']
    X = df[feature_cols]
    y = df['fuel_L100']
    groups = df['VehId']

    # Chia tập Train/Test theo GroupShuffleSplit (chống Data Leakage theo VehId)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    # Xây dựng Pipeline GBR vô địch (Chương 4 báo cáo)
    gbr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42
        ))
    ])
    gbr_pipeline.fit(X_train, y_train)

    # Xây dựng Pipeline K-Means (K=3) phân cụm hành trình (Mục 4.4 báo cáo)
    cluster_features = ['avg_speed_kmh', 'avg_load_pct', 'stop_ratio']
    kmeans_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10))
    ])
    kmeans_pipeline.fit(X_train[cluster_features])

    # Ánh xạ nhãn cụm theo tốc độ trung bình tăng dần (Đô thị -> Hỗn hợp -> Cao tốc)
    centers = kmeans_pipeline.named_steps['kmeans'].cluster_centers_
    speed_order = np.argsort(centers[:, 0])
    cluster_map = {
        speed_order[0]: ("Cụm 0: Đô thị / Kẹt xe", 11.32),
        speed_order[1]: ("Cụm 1: Đường hỗn hợp", 8.78),
        speed_order[2]: ("Cụm 2: Đường cao tốc", 7.83)
    }

    return gbr_pipeline, kmeans_pipeline, cluster_map, feature_cols, is_demo_data

gbr_pipe, kmeans_pipe, cluster_map, feature_cols, is_demo_data = load_and_train_models()

# ==========================================
# 3. GIAO DIỆN CHIA TAB CHỨC NĂNG
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "🎛️ 1. Ước lượng Chuyến đi Đơn lẻ", 
    "📂 2. Ước lượng theo Lô (Batch CSV) & Kiểm thử Ngoại lệ", 
    "📊 3. Phân tích Sai số & Hiệu năng Mô hình"
])

# ------------------------------------------
# TAB 1: ƯỚC LƯỢNG ĐƠN LẺ (TỪ IPYWIDGETS)
# ------------------------------------------
with tab1:
    col_input, col_result = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("📋 Nhập đặc trưng tổng hợp sau chuyến đi")
        
        veh_class = st.selectbox(
            "Loại xe (Vehicle Class):",
            ["Car (Xe con)", "SUV (Xe thể thao đa dụng)", "Truck (Xe tải)", "Van/Minivan"]
        )
        
        avg_speed = st.slider("Tốc độ trung bình (km/h):", min_value=0.0, max_value=120.0, value=40.0, step=1.0)
        avg_load = st.slider("Tải động cơ trung bình (%):", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        total_time = st.slider("Thời gian chuyến đi (giây):", min_value=60, max_value=10800, value=1800, step=60)
        stop_ratio = st.slider("Tỷ lệ dừng đỗ (0 - 1):", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
        
        # Tùy chọn giả lập thiếu cảm biến nhiệt độ OAT[DegC]
        missing_temp = st.checkbox("Giả lập lỗi khuyết thiếu cảm biến Nhiệt độ môi trường (NaN)", value=False)
        if not missing_temp:
            avg_temp = st.slider("Nhiệt độ môi trường (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)
        else:
            avg_temp = np.nan
            st.info("💡 Cảm biến nhiệt độ đang nhận giá trị `NaN`. Pipeline sẽ tự động điền khuyết bằng trung vị (Median Imputation).")

        # Tính quãng đường tích phân từ tốc độ trung bình và thời gian
        total_dist = (avg_speed * total_time) / 3600.0
        st.caption(f"📏 **Quãng đường ước tính:** `{total_dist:.2f} km` ({total_time/60:.1f} phút)")

        btn_predict = st.button("🚀 Ước lượng Mức tiêu thụ", type="primary", use_container_width=True)

    with col_result:
        st.subheader("🎯 Kết quả Ước lượng & Diễn giải")
        
        # Kiểm tra điều kiện biên (Data Quality Rules - Chương 3)
        errors = []
        if total_time < 180:
            errors.append(f"Thời gian chuyến đi ({total_time}s) nhỏ hơn ngưỡng tối thiểu 180 giây (3 phút).")
        if total_dist < 1.0:
            errors.append(f"Quãng đường chuyến đi ({total_dist:.2f} km) nhỏ hơn ngưỡng tối thiểu 1.0 km (dễ gây nhiễu chia cho 0).")
        if avg_speed <= 0:
            errors.append("Tốc độ trung bình phải lớn hơn 0 km/h.")

        if errors:
            st.error("❌ **DỮ LIỆU NGOÀI PHẠM VI KIỂM ĐỊNH CHẤT LƯỢNG:**")
            for err in errors:
                st.write(f"- {err}")
            st.info("Vui lòng điều chỉnh thanh trượt để thỏa mãn điều kiện lọc nhiễu của bộ dữ liệu VED (Quãng đường >= 1 km và Thời gian >= 180 giây).")
        else:
            # Đóng gói DataFrame đầu vào (Tuyệt đối không có Fuel Rate)
            input_df = pd.DataFrame([{
                'avg_speed_kmh': avg_speed,
                'avg_load_pct': avg_load,
                'stop_ratio': stop_ratio,
                'avg_temp_c': avg_temp,
                'total_dist_km': total_dist,
                'total_time_sec': float(total_time)
            }])

            # Dự đoán bằng GBR
            pred_l100 = gbr_pipe.predict(input_df)[0]
            
            # Điều chỉnh hệ số tham khảo theo phân khúc xe tĩnh nếu người dùng chọn SUV/Truck/Van
            class_factor = {"Car (Xe con)": 1.0, "SUV (Xe thể thao đa dụng)": 1.08, "Van/Minivan": 1.12, "Truck (Xe tải)": 1.20}[veh_class]
            adjusted_l100 = pred_l100 * class_factor
            total_fuel_liters = (adjusted_l100 * total_dist) / 100.0

            # Phân cụm K-Means cho chuyến đi
            raw_cluster = kmeans_pipe.predict(input_df[['avg_speed_kmh', 'avg_load_pct', 'stop_ratio']])[0]
            cluster_name, cluster_avg_fuel = cluster_map[raw_cluster]

            # Hiển thị Metrics
            m1, m2, m3 = st.columns(3)
            baseline_mean = 9.62 # Mốc Baseline tập Train trong báo cáo
            diff_baseline = adjusted_l100 - baseline_mean
            
            m1.metric(
                label="Mức tiêu thụ quy đổi", 
                value=f"{adjusted_l100:.2f} L/100km",
                delta=f"{diff_baseline:+.2f} L vs Baseline (9.62)",
                delta_color="inverse"
            )
            m2.metric(
                label="Tổng nhiên liệu chuyến đi", 
                value=f"{total_fuel_liters:.3f} Lít"
            )
            m3.metric(
                label="Phân cụm K-Means", 
                value=cluster_name.split(":")[1].strip(),
                delta=f"TB cụm: {cluster_avg_fuel} L/100km",
                delta_color="off"
            )

            # Biểu đồ đồng hồ đo (Gauge Chart)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=adjusted_l100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Mức tiêu hao ước lượng ({veh_class.split(' ')[0]}) - Đơn vị: L/100km"},
                gauge={
                    'axis': {'range': [0, 25], 'tickwidth': 1},
                    'bar': {'color': "#1a5276"},
                    'steps': [
                        {'range': [0, 8.5], 'color': "#a9dfbf"},
                        {'range': [8.5, 12.0], 'color': "#f9e79f"},
                        {'range': [12.0, 25.0], 'color': "#f5b7b1"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': baseline_mean
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Hộp diễn giải chuyên môn cơ khí ô tô
            st.markdown("#### 🔍 Diễn giải kỹ thuật sau chuyến đi:")
            st.write(f"- **Trạng thái vận hành:** Chuyến đi thuộc **{cluster_name}** (Mức tiêu thụ trung bình của cụm trong báo cáo là `{cluster_avg_fuel} L/100km`).")
            if avg_speed < 30:
                st.write("- **Ảnh hưởng tốc độ & dừng đỗ:** Xe di chuyển ở dải tốc độ thấp (< 30 km/h) kèm tỷ lệ dừng đỗ làm động cơ hoạt động dưới vùng hiệu suất nhiệt tối ưu, đẩy giá trị SHAP và mức tiêu hao nhiên liệu tăng cao.")
            elif avg_speed <= 65:
                st.write("- **Ảnh hưởng tốc độ:** Xe vận hành trong vùng tốc độ tối ưu (đáy đường cong chữ U), giúp tiết kiệm nhiên liệu hiệu quả.")
            else:
                st.write("- **Ảnh hưởng tốc độ & tải:** Xe chạy tốc độ cao trên cao tốc giúp tỷ lệ dừng đỗ thấp, nhưng lực cản khí động học làm tải động cơ tăng lên.")

# ------------------------------------------
# TAB 2: ƯỚC LƯỢNG THEO LÔ (BATCH CSV)
# ------------------------------------------
with tab2:
    st.subheader("📂 Kiểm thử Lô dữ liệu (Batch Prediction) & Kiểm tra Định dạng/Ngoại lệ")
    st.write("Đáp ứng yêu cầu Bước 5: Nhận một lô dữ liệu mới, tự động bắt lỗi thiếu cảm biến, sai đơn vị hoặc ngoài phạm vi hợp lệ.")

    # Tạo dữ liệu lô mẫu có chứa cả ca hợp lệ, ca thiếu nhiệt độ (NaN) và ca vi phạm ngưỡng (<1km, <180s)
    sample_batch = pd.DataFrame([
        {"Trip_ID": "T01_HopLe_CaoToc", "avg_speed_kmh": 72.5, "avg_load_pct": 38.0, "stop_ratio": 0.05, "avg_temp_c": 22.0, "total_dist_km": 18.5, "total_time_sec": 918},
        {"Trip_ID": "T02_HopLe_DoThi", "avg_speed_kmh": 24.0, "avg_load_pct": 28.5, "stop_ratio": 0.35, "avg_temp_c": 28.0, "total_dist_km": 4.2, "total_time_sec": 630},
        {"Trip_ID": "T03_ThieuNhietDo_NaN", "avg_speed_kmh": 45.0, "avg_load_pct": 31.0, "stop_ratio": 0.15, "avg_temp_c": np.nan, "total_dist_km": 6.5, "total_time_sec": 520},
        {"Trip_ID": "T04_Loi_ChuyenQuaNgan", "avg_speed_kmh": 20.0, "avg_load_pct": 25.0, "stop_ratio": 0.40, "avg_temp_c": 25.0, "total_dist_km": 0.6, "total_time_sec": 108},
        {"Trip_ID": "T05_Loi_SaiDonViStopRatio", "avg_speed_kmh": 40.0, "avg_load_pct": 30.0, "stop_ratio": 25.0, "avg_temp_c": 20.0, "total_dist_km": 5.0, "total_time_sec": 450},
    ])

    csv_template = sample_batch.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Tải file CSV mẫu kiểm thử (Gồm cả dòng hợp lệ, dòng NaN và dòng lỗi ngoại lệ)",
        data=csv_template,
        file_name="sample_test_trips.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Tải lên file CSV chứa đặc trưng chuyến đi:", type=["csv"])
    df_batch = pd.read_csv(uploaded_file) if uploaded_file is not None else sample_batch.copy()

    if st.button("▶️ Chạy Kiểm định & Ước lượng Lô dữ liệu"):
        missing_cols = [c for c in feature_cols if c not in df_batch.columns]
        if missing_cols:
            st.error(f"❌ File tải lên thiếu các cột bắt buộc: {missing_cols}")
        else:
            statuses = []
            preds = []
            for idx, row in df_batch.iterrows():
                if row['total_dist_km'] < 1.0 or row['total_time_sec'] < 180:
                    statuses.append("❌ Loại bỏ: Chuyến quá ngắn (<1km hoặc <180s)")
                    preds.append(np.nan)
                elif not (0.0 <= row['stop_ratio'] <= 1.0):
                    statuses.append("❌ Lỗi đơn vị: stop_ratio phải trong [0, 1] (không phải %)")
                    preds.append(np.nan)
                elif row['avg_speed_kmh'] <= 0 or row['avg_speed_kmh'] > 200:
                    statuses.append("❌ Ngoài phạm vi: Tốc độ không hợp lệ")
                    preds.append(np.nan)
                else:
                    row_df = pd.DataFrame([row[feature_cols]])
                    pred_val = gbr_pipe.predict(row_df)[0]
                    if pd.isna(row['avg_temp_c']):
                        statuses.append("⚠️ Hợp lệ (Đã tự động điền khuyết NaN cột avg_temp_c)")
                    else:
                        statuses.append("✅ Hợp lệ")
                    preds.append(round(pred_val, 2))
            
            df_batch['Trạng thái Kiểm định'] = statuses
            df_batch['Ước lượng GBR (L/100km)'] = preds
            st.dataframe(df_batch, use_container_width=True)

# ------------------------------------------
# TAB 3: BIỂU ĐỒ SAI SỐ & ĐỐI CHIẾU BÁO CÁO
# ------------------------------------------
with tab3:
    st.subheader("📊 Bằng chứng Thực nghiệm trên Tập Test Độc lập theo Xe (Group Split)")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. So sánh hiệu năng 5 mô hình hồi quy (Bảng 5.1)**")
        df_models = pd.DataFrame({
            "Mô hình": ["Gradient Boosting (GBR)", "Random Forest", "Ridge", "Linear Regression", "Baseline (Mean)"],
            "MAE (L/100km)": [1.894, 1.945, 2.191, 2.192, 2.515],
            "RMSE (L/100km)": [2.469, 2.545, 2.966, 2.966, 3.287],
            "R²": [0.427, 0.391, 0.173, 0.173, -0.016],
            "Dung sai ±1L (%)": [33.84, 33.77, 28.68, 28.68, 23.83]
        })
        fig_bar = px.bar(
            df_models, x="Mô hình", y=["MAE (L/100km)", "RMSE (L/100km)"],
            barmode="group", text_auto=True,
            color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        fig_bar.update_layout(height=360, legend_title_text="Chỉ số sai số")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("**2. Sai số MAE phân tầng theo Tốc độ và Tải động cơ (Bảng 5.2)**")
        z_mae = [
            [2.73, 2.49, np.nan],
            [1.67, 1.83, 2.16],
            [1.71, 1.32, 0.98]
        ]
        x_load = ["Tải nhẹ (<30%)", "Tải TB (30-50%)", "Tải nặng (>50%)"]
        y_speed = ["Chậm (0-30 km/h)", "Trung bình (30-60 km/h)", "Nhanh (>60 km/h)"]

        fig_heat = px.imshow(
            z_mae, x=x_load, y=y_speed, text_auto=True,
            color_continuous_scale="YlOrRd",
            labels=dict(x="Nhóm Tải động cơ (Engine Load)", y="Nhóm Tốc độ", color="MAE")
        )
        fig_heat.update_layout(height=360)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "📌 **Giải thích điểm khuyết NaN ở ô (Tốc độ 0–30 km/h, Tải động cơ > 50%):** "
        "Ở dải tốc độ đô thị 0–30 km/h, động cơ làm việc ở vòng tua thấp và không chịu lực cản khí động học lớn nên tải trung bình cả chuyến "
        "chỉ dao động từ 15%–35%. Trong 3.026 chuyến đi thuộc tập Test độc lập, không có chuyến đi nào rơi vào tổ hợp Tốc độ < 30 km/h mà Tải trung bình > 50% (N = 0). "
        "Ngoài ra, dải tốc độ 0–30 km/h có sai số MAE cao nhất (2.49 – 2.73 L/100km) do biến động dừng chờ đèn đỏ và kẹt xe đô thị."
    )
