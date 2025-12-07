import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG (Page Config)
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Trợ lý Tim Mạch AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# 2. HÀM LOAD DỮ LIỆU VÀ MODEL
# -------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load model đã huấn luyện từ file .joblib"""
    try:
        return joblib.load('cardio_model.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Load dữ liệu gốc để vẽ biểu đồ so sánh"""
    try:
        df = pd.read_csv('cardio_train.csv', sep=';')
        # Feature Engineering cho dữ liệu hiển thị
        df['age_years'] = (df['age'] / 365.25).round(1)
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        return df
    except FileNotFoundError:
        return None

# Gọi hàm load
model = load_model()
df_data = load_data()

# -------------------------------------------------------------------------
# 3. LOGIC BÁC SĨ ẢO (Rule-based System)
# -------------------------------------------------------------------------
def give_advice(bmi, ap_hi, ap_lo, smoke, alco, active, prob):
    """
    Hàm trả về danh sách lời khuyên dựa trên chỉ số sức khỏe.
    """
    advice_list = []
    
    # 1. Đánh giá BMI
    if bmi < 18.5:
        advice_list.append("⚠️ **Cân nặng:** Bạn hơi gầy. Cần bổ sung dinh dưỡng.")
    elif 18.5 <= bmi < 24.9:
        advice_list.append("✅ **Cân nặng:** Tuyệt vời! BMI ở mức chuẩn.")
    elif 25 <= bmi < 29.9:
        advice_list.append("⚠️ **Cân nặng:** Bạn đang **Thừa cân**. Nên giảm tinh bột, tăng rau xanh.")
    else:
        advice_list.append("🚨 **Cân nặng:** Bạn đang **Béo phì**. Nguy cơ tim mạch cao. Cần giảm cân ngay.")

    # 2. Đánh giá Huyết áp
    if ap_hi >= 140 or ap_lo >= 90:
        advice_list.append("🚨 **Huyết áp:** Bạn bị **Cao huyết áp**. Hãy hạn chế ăn mặn và đi khám bác sĩ.")
    elif ap_hi >= 130 or ap_lo >= 80:
        advice_list.append("⚠️ **Huyết áp:** Huyết áp hơi cao (Tiền tăng huyết áp). Cần theo dõi thường xuyên.")
    else:
        advice_list.append("✅ **Huyết áp:** Huyết áp ổn định.")

    # 3. Đánh giá Lối sống
    if smoke == 1:
        advice_list.append("🚭 **Hút thuốc:** Ngừng hút thuốc ngay để giảm 50% nguy cơ đột quỵ.")
    if alco == 1:
        advice_list.append("🍷 **Rượu bia:** Hạn chế rượu bia để bảo vệ gan và tim.")
    if active == 0:
        advice_list.append("🏃 **Vận động:** Bạn ít vận động. Hãy đi bộ ít nhất 30 phút/ngày.")

    # 4. Lời khuyên tổng quan từ AI
    if prob > 0.7:
        advice_list.append("🏥 **CẢNH BÁO KHẨN:** AI dự báo nguy cơ > 70%. Bạn nên đi tầm soát tim mạch tại bệnh viện sớm.")
    
    return advice_list

# -------------------------------------------------------------------------
# 4. GIAO DIỆN: SIDEBAR
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
# 5. GIAO DIỆN CHÍNH
# -------------------------------------------------------------------------
st.title("❤️ Hệ thống Dự đoán & Tư vấn Tim Mạch")
st.markdown("---")

# Chia layout 2 cột: Trái (Input) - Phải (Kết quả & Biểu đồ)
col_input, col_output = st.columns([1, 1.5], gap="large")

# --- CỘT TRÁI: FORM NHẬP LIỆU ---
with col_input:
    st.subheader("📝 Nhập chỉ số sức khỏe")
    with st.form("input_form"):
        # Thông tin cơ bản
        age_input = st.number_input("Tuổi", 1, 100, 50)
        gender_input = st.selectbox("Giới tính", [1, 2], format_func=lambda x: "Nữ" if x==1 else "Nam")
        
        c1, c2 = st.columns(2)
        with c1: height_input = st.number_input("Chiều cao (cm)", 100, 250, 165)
        with c2: weight_input = st.number_input("Cân nặng (kg)", 30.0, 200.0, 65.0)
            
        # Chỉ số y khoa
        st.markdown("**Chỉ số y khoa:**")
        ap_hi_input = st.number_input("Huyết áp tâm thu (Trên)", 60, 240, 120)
        ap_lo_input = st.number_input("Huyết áp tâm trương (Dưới)", 40, 160, 80)
        
        cholesterol_input = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Bình thường", "Cao", "Rất cao"][x-1])
        gluc_input = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Bình thường", "Cao", "Rất cao"][x-1])
        
        # Lối sống
        st.markdown("**Lối sống:**")
        check1, check2, check3 = st.columns(3)
        with check1: smoke_input = st.checkbox("Hút thuốc")
        with check2: alco_input = st.checkbox("Rượu bia")
        with check3: active_input = st.checkbox("Thể thao")
        
        st.markdown("---")
        submit_btn = st.form_submit_button("🔍 PHÂN TÍCH NGAY", type="primary")

# --- CỘT PHẢI: XỬ LÝ & HIỂN THỊ ---
with col_output:
    # Biến lưu trạng thái để vẽ biểu đồ
    user_bmi = 0
    
    if submit_btn and model:
        # 1. Xử lý dữ liệu đầu vào
        user_bmi = weight_input / ((height_input/100)**2)
        
        # Tạo DataFrame đúng chuẩn input của model
        input_data = pd.DataFrame([[
            gender_input, height_input, weight_input, ap_hi_input, ap_lo_input, 
            cholesterol_input, gluc_input, 
            1 if smoke_input else 0, 
            1 if alco_input else 0, 
            1 if active_input else 0, 
            age_input, user_bmi
        ]], columns=['age_years','gender', 'height', 'weight', 'bmi' ,'ap_hi', 'ap_lo', 
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
        
        # 2. Dự đoán bằng AI
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # 3. Hiển thị Kết quả
        st.subheader("📊 Kết quả Phân tích")
        
        # Thanh đo mức độ rủi ro
        st.write(f"Tỉ lệ nguy cơ tim mạch: **{probability*100:.1f}%**")
        bar_color = "red" if probability > 0.5 else "green"
        st.progress(int(probability*100))
        
        if prediction == 1:
            st.error(f"⚠️ **CẢNH BÁO:** Bạn CÓ nguy cơ mắc bệnh tim mạch.")
        else:
            st.success(f"✅ **AN TOÀN:** Bạn ít có nguy cơ mắc bệnh.")
            
        # 4. Hiển thị Lời khuyên (Bác sĩ ảo)
        st.subheader("💡 Lời khuyên cá nhân hóa")
        with st.expander("Xem chi tiết lời khuyên từ chuyên gia", expanded=True):
            advice = give_advice(user_bmi, ap_hi_input, ap_lo_input, 
                                 1 if smoke_input else 0, 
                                 1 if alco_input else 0, 
                                 1 if active_input else 0, 
                                 probability)
            for item in advice:
                st.write(item)

    # --- PHẦN BIỂU ĐỒ (DASHBOARD) ---
    # Luôn hiển thị nếu có Data, không cần chờ nút bấm để làm đẹp giao diện ban đầu
    if df_data is not None:
        st.markdown("---")
        st.subheader("📈 Biểu đồ so sánh cộng đồng")
        
        tab1, tab2, tab3 = st.tabs(["Vị trí của bạn", "Phân bố BMI", "Tương quan"])
        
        # Tab 1: Scatter Plot (Điểm nhấn)
        with tab1:
            st.caption("So sánh chỉ số của bạn với 500 người ngẫu nhiên.")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            
            # Vẽ nền (500 người)
            sample_df = df_data.sample(500)
            sns.scatterplot(data=sample_df, x='age_years', y='ap_hi', hue='cardio', 
                            palette={0:'green', 1:'red'}, alpha=0.3, ax=ax1)
            
            # Vẽ điểm người dùng (Nếu đã nhập liệu)
            if submit_btn:
                ax1.scatter(age_input, ap_hi_input, color='blue', s=300, marker='*', label='BẠN')
                ax1.legend()
                
            plt.xlabel("Tuổi")
            plt.ylabel("Huyết áp tâm thu")
            st.pyplot(fig1)

        # Tab 2: Histogram BMI
        with tab2:
            st.caption("Phân bố chỉ số BMI trong cộng đồng.")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df_data, x='bmi', hue='cardio', kde=True, 
                         palette={0:'green', 1:'red'}, ax=ax2)
            
            # Vẽ vạch BMI của user
            if submit_btn:
                plt.axvline(user_bmi, color='blue', linestyle='--', label=f'BMI của bạn ({user_bmi:.1f})')
                plt.legend()
                
            plt.xlim(15, 45)
            st.pyplot(fig2)

        # Tab 3: Heatmap
        with tab3:
            st.caption("Mức độ ảnh hưởng của các yếu tố.")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            cols = ['age_years', 'ap_hi', 'weight', 'bmi', 'cholesterol', 'cardio']
            sns.heatmap(df_data[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
            st.pyplot(fig3)
    
    elif not model:
        # Màn hình chờ khi chưa có model
        st.info("👈 Vui lòng upload file 'cardio_model.joblib' để bắt đầu.")
        st.image("https://img.freepik.com/free-vector/medical-technology-science-background-vector-blue-tone_53876-119567.jpg", use_column_width=True)
