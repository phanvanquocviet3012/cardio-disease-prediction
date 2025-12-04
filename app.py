import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG (Phải nằm đầu tiên)
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Dự đoán Tim Mạch",
    page_icon="❤️",
    layout="wide"  # Dùng layout rộng để hiển thị biểu đồ đẹp hơn
)

# -------------------------------------------------------------------------
# 2. LOAD DỮ LIỆU & MODEL
# -------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Load model đã train (đảm bảo file này nằm cùng thư mục)
        return joblib.load('cardio_model.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        # Load data gốc để vẽ biểu đồ
        df = pd.read_csv('cardio_train.csv', sep=';')
        # Tạo thêm cột age_years và bmi để phân tích
        df['age_years'] = (df['age'] / 365.25).round(1)
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        return df
    except FileNotFoundError:
        return None

model = load_model()
df_data = load_data()

# -------------------------------------------------------------------------
# 3. SIDEBAR: THÔNG TIN NHÓM
# -------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.title("Cardio Prediction")
    st.info(
        """
        **Đồ án Machine Learning**
        
        Thành viên nhóm:
        1. Nguyễn Khánh Vân (Model)
        2. Phan Văn Quốc Việt (App)
        3. Đỗ Hải Yến (Data)
        
        GVHD: TS. Võ Nguyễn Lê Duy
        """
    )
    st.divider()
    if model is not None:
        st.success("✅ Model: Đã tải thành công")
    else:
        st.error("❌ Model: Chưa tìm thấy file .joblib")
        
    if df_data is not None:
        st.success("✅ Data: Đã tải thành công")
    else:
        st.error("❌ Data: Chưa tìm thấy file .csv")

# -------------------------------------------------------------------------
# 4. GIAO DIỆN CHÍNH
# -------------------------------------------------------------------------
st.title("❤️ Hệ thống Dự đoán Nguy cơ Bệnh Tim Mạch")
st.markdown("Nhập các chỉ số sức khỏe để dự đoán nguy cơ và xem phân tích trực quan.")
st.divider()

# Chia layout thành 2 cột: Cột trái (Nhập liệu) - Cột phải (Kết quả & Biểu đồ)
col_input, col_viz = st.columns([1, 2]) # Tỉ lệ 1:2

# --- CỘT TRÁI: FORM NHẬP LIỆU ---
with col_input:
    st.subheader("📝 Nhập thông tin")
    with st.form("prediction_form"):
        age_years = st.number_input("Tuổi", 10, 100, 50)
        gender = st.selectbox("Giới tính", [1, 2], format_func=lambda x: "Nữ" if x==1 else "Nam")
        
        c1, c2 = st.columns(2)
        with c1:
            height = st.number_input("Chiều cao (cm)", 100, 250, 165)
        with c2:
            weight = st.number_input("Cân nặng (kg)", 30.0, 200.0, 65.0)
            
        ap_hi = st.number_input("Huyết áp tâm thu (Trên)", 60, 240, 120)
        ap_lo = st.number_input("Huyết áp tâm trương (Dưới)", 40, 160, 80)
        
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Bình thường", "Cao", "Rất cao"][x-1])
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Bình thường", "Cao", "Rất cao"][x-1])
        
        st.markdown("---")
        check1, check2, check3 = st.columns(3)
        with check1: smoke = st.checkbox("Hút thuốc")
        with check2: alco = st.checkbox("Rượu bia")
        with check3: active = st.checkbox("Thể thao")
        
        submit_btn = st.form_submit_button("🔍 DỰ ĐOÁN NGAY", type="primary")

# --- XỬ LÝ DỰ ĐOÁN ---
prediction_result = None
prob = 0.0
bmi_user = 0.0

if submit_btn:
    if model:
        # Tính toán BMI
        bmi_user = weight / ((height/100)**2)
        
        # Tạo dataframe input đúng chuẩn model yêu cầu
        input_data = pd.DataFrame([[
            age_years, gender, height, weight, bmi_user, ap_hi, ap_lo, 
            cholesterol, gluc, 
            1 if smoke else 0, 
            1 if alco else 0, 
            1 if active else 0
        ]], columns=['age_years','gender', 'height', 'weight', 'bmi' ,'ap_hi', 'ap_lo', 
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
        
        # Dự đoán
        prediction_result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        st.error("Lỗi: Không tìm thấy model để dự đoán!")

# --- CỘT PHẢI: KẾT QUẢ & BIỂU ĐỒ ---
with col_viz:
    # 1. Hiển thị kết quả dự đoán (nếu đã bấm nút)
    if submit_btn and prediction_result is not None:
        st.subheader("📋 Kết quả dự đoán")
        if prediction_result == 1:
            st.error(f"⚠️ **CẢNH BÁO:** Bạn có nguy cơ mắc bệnh tim mạch! (Tỉ lệ: {prob*100:.1f}%)")
            st.write(f"Chỉ số BMI của bạn: **{bmi_user:.1f}**")
        else:
            st.success(f"✅ **AN TOÀN:** Bạn ít có nguy cơ mắc bệnh. (Tỉ lệ: {prob*100:.1f}%)")
            st.write(f"Chỉ số BMI của bạn: **{bmi_user:.1f}**")
        st.divider()

    # 2. Phần trực quan hóa (Dashboard)
    st.subheader("📊 Phân tích dữ liệu & So sánh")
    
    if df_data is not None:
        tab1, tab2, tab3 = st.tabs(["Tương quan (Heatmap)", "Phân bố Tuổi", "Vị trí của bạn"])
        
        # Tab 1: Heatmap
        with tab1:
            st.write("Mức độ ảnh hưởng của các chỉ số đến bệnh tim (Màu đỏ càng đậm càng nguy hiểm).")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            cols = ['age_years', 'ap_hi', 'weight', 'bmi', 'cholesterol', 'cardio']
            sns.heatmap(df_data[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
            st.pyplot(fig1)

        # Tab 2: Histogram
        with tab2:
            st.write("Độ tuổi nào dễ mắc bệnh nhất?")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df_data, x='age_years', hue='cardio', kde=True, palette={0:'green', 1:'red'}, ax=ax2)
            plt.legend(['Có bệnh', 'Không bệnh'])
            st.pyplot(fig2)

        # Tab 3: Scatter Plot (So sánh user với data)
        with tab3:
            st.write("Bạn đang ở đâu so với 500 người ngẫu nhiên trong dữ liệu?")
            if submit_btn: # Chỉ hiện điểm đỏ khi user đã nhập liệu
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                # Lấy mẫu 500 người
                sample = df_data.sample(500)
                sns.scatterplot(data=sample, x='age_years', y='ap_hi', hue='cardio', palette={0:'green', 1:'red'}, alpha=0.5, ax=ax3)
                
                # Vẽ điểm của User
                ax3.scatter(age_years, ap_hi, color='blue', s=300, marker='*', label='BẠN')
                plt.xlabel("Tuổi")
                plt.ylabel("Huyết áp tâm thu")
                plt.legend()
                st.pyplot(fig3)
            else:
                st.info("Hãy nhập thông tin và bấm 'Dự đoán' để xem vị trí của bạn trên biểu đồ.")
    else:
        st.warning("Đang chờ file 'cardio_train.csv' để vẽ biểu đồ...")

# Footer
st.markdown("---")
st.caption("Developed with Streamlit by Team 3")
