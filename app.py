import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# 1. Cấu hình trang (Luôn phải nằm đầu tiên)
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Dự đoán Tim Mạch - Demo",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------------------------------------------------
# 2. Sidebar: Thông tin nhóm
# -------------------------------------------------------------------------
st.sidebar.title("Thông tin nhóm")
st.sidebar.info(
    """
    **Thành viên:**
    1. Thành viên A (Model)
    2. Thành viên B (App)
    3. Thành viên C (Data)
    
    **Giảng viên:** [Tên Giảng Viên]
    """
)

st.sidebar.header("Trạng thái hệ thống")
st.sidebar.success("✅ Server đang chạy ổn định")

# -------------------------------------------------------------------------
# 3. Giao diện chính (Main Layout)
# -------------------------------------------------------------------------
st.title("🏥 Hệ thống Dự đoán Nguy cơ Bệnh Tim")
st.markdown("---")

# Thông báo trạng thái (Placeholder)
st.warning("⚠️ **LƯU Ý:** Đây là phiên bản thử nghiệm giao diện (Prototype). Model dự đoán chưa được tích hợp.")

# Demo form nhập liệu (Chỉ để test giao diện, chưa xử lý logic)
st.subheader("Nhập thông tin bệnh nhân (Demo)")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    height = st.number_input("Chiều cao (cm)", 100, 250, 165)

with col2:
    weight = st.number_input("Cân nặng (kg)", 30.0, 200.0, 60.0)
    ap_hi = st.number_input("Huyết áp tâm thu", 60, 240, 120)
    ap_lo = st.number_input("Huyết áp tâm trương", 40, 160, 80)

# Nút bấm thử nghiệm
if st.button("🔍 Chạy thử dự đoán"):
    st.balloons()  # Hiệu ứng bóng bay để biết code đã chạy
    st.info(f"Dữ liệu đã nhận: {age} tuổi, {gender}, cao {height}cm, nặng {weight}kg.")
    st.write("🔜 Kết quả dự đoán sẽ hiện ở đây khi Model được tích hợp.")

# -------------------------------------------------------------------------
# 4. Footer
# -------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2024 - Đồ án Machine Learning nhóm 3 người.")
