# Cardiological Disease Prediction

Dự án sử dụng Machine Learning để dự đoán nguy cơ mắc bệnh tim mạch dựa trên các chỉ số sinh học và thói quen sinh hoạt của người bệnh.

## 📌 Giới thiệu

Mục tiêu của dự án là xây dựng một mô hình phân loại (Classification) có độ chính xác cao để hỗ trợ các bác sĩ trong việc chẩn đoán sớm bệnh tim.

* [Cardio Prediction](https://cardiodisease-prediction.streamlit.app/)

## 📊 Dữ liệu

DATASET: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)

Dữ liệu được sử dụng trong dự án bao gồm các đặc trưng chính như:

* **Age**: Tuổi.
* **Gender**: Giới tính.
* **Systolic/Diastolic BP**: Huyết áp tâm thu và tâm trương.
* **Cholesterol**: Mức cholesterol trong máu.
* **Smoke/Alco**: Thói quen hút thuốc/uống rượu bia.
* **Active**: Mức độ vận động thể chất.

## 🛠 Công nghệ sử dụng

* **Ngôn ngữ:** Python
* **Thư viện chính:**
* `pandas`, `numpy`: Xử lý dữ liệu.
* `scikit-learn`: Xây dựng mô hình (Random Forest, Logistic Regression, v.v.).
* `seaborn`, `matplotlib`: Trực quan hóa dữ liệu.
* `jupyter notebook`: Môi trường phát triển.



## 🚀 Cài đặt và Sử dụng

### 1. Clone repository

```bash
git clone https://github.com/phanvanquocviet3012/cardio-disease-prediction.git
cd cardio-disease-prediction

```

### 2. Cài đặt môi trường

Khuyến khích sử dụng môi trường ảo:

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Chạy dự án

Mở file Notebook để xem chi tiết quá trình huấn luyện:

```bash
streamlit run app.py

```

## 📈 Kết quả

* **Mô hình tốt nhất:** Random Forest Classifier.
* **Độ chính xác (Accuracy):** ~85% (Thay đổi số liệu thực tế của bạn).
* **F1-Score:** 0.82.

## 👤 Tác giả
* **Phan Văn Quốc Việt**
* **Nguyễn Khánh Vân**
* **Đỗ Hải Yến**
