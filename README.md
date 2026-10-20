# SmartHVACControlByDoubleQLearning

Ứng dụng thuật toán **Double Q-Learning** trong điều khiển thông minh hệ thống HVAC (Heating, Ventilation, and Air Conditioning) cho tòa nhà thông minh, sử dụng bộ dữ liệu thực tế **CU-BEMS (Smart Building Energy and IAQ Data)**.

---

## 🌟 Tổng Quan Dự Án

Dự án phát triển một hệ thống điều khiển thông minh sử dụng **Học tăng cường (Reinforcement Learning)**, cụ thể là thuật toán **Double Q-Learning**, nhằm đưa ra các quyết định điều chỉnh công suất làm mát của hệ thống điều hòa HVAC dựa trên thông số môi trường thực tế (Nhiệt độ, Độ ẩm, Ánh sáng, Phụ tải máy tính/ổ cắm).

Mục tiêu cốt lõi là giải bài toán đa mục tiêu:
1. **Tiện nghi nhiệt (Thermal Comfort):** Giữ nhiệt độ phòng ổn định trong dải tiêu chuẩn **ASHRAE Standard 55** ($22.0^\circ\text{C} - 24.5^\circ\text{C}$).
2. **Tiết kiệm năng lượng (Energy Conservation):** Tối ưu hóa điện năng tiêu thụ, tránh làm lạnh quá mức (overcooling) và giảm tải vào các khung giờ cao điểm.

---

## 📊 Bộ Dữ Liệu: CU-BEMS
- **Nguồn:** [CU-BEMS – Smart Building Energy and IAQ Data (Kaggle)](https://www.kaggle.com/datasets/claytonmiller/cubems-smart-building-energy-and-iaq-data).
- **Đặc điểm:** Dữ liệu chuỗi thời gian thu thập từ 7 tầng tòa nhà Đại học Chulalongkorn, bao gồm cảm biến môi trường IAQ (Nhiệt độ, Độ ẩm, Độ rọi ánh sáng) và hệ thống đo đếm điện năng chi tiết từng mạch điều hòa (AC) và phụ tải ổ cắm (Plug).

---

## 🏗️ Cấu Trúc Dự Án

```text
SmartHVACControlByDoubleQLearning/
├── app/
│   └── app.py                     # Ứng dụng Web tương tác Streamlit
├── data/
│   ├── processed/                 # Dữ liệu sạch chu kỳ 1 giờ và 15 phút
│   └── sample/                    # Mẫu dữ liệu thử nghiệm nhanh
├── doc/
│   └── TECHNICAL_REPORT.md        # Báo cáo kỹ thuật chi tiết
├── model/
│   ├── q_learning_model.pkl       # Trọng số Q-Table của Q-Learning (Baseline)
│   └── double_q_learning_model.pkl# Trọng số Q-Table của Double Q-Learning (Proposed)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb        # Khám phá và trực quan hóa dữ liệu (EDA)
│   ├── 02_preprocessing_and_environment.ipynb # Mô hình nhiệt và môi trường RL
│   └── 03_training_and_comparison.ipynb     # Huấn luyện và đánh giá so sánh
├── result/
│   ├── figure/                    # Đồ thị EDA, quá trình hội tụ, so sánh hiệu năng
│   └── log/                       # Nhật ký huấn luyện và bảng số liệu kiểm thử
├── src/
│   ├── agent/                     # Q-Learning, Double Q-Learning, Rule-Based
│   ├── data/                      # Loader, Preprocessing, Rời rạc hóa State
│   ├── environment/               # RC Building Thermal Model, SmartHVACEnv
│   ├── evaluation/                # Bộ công cụ đánh giá benchmark
│   ├── reward/                    # Hàm phần thưởng đa mục tiêu
│   └── util/                      # Tiện ích vẽ đồ thị trực quan hóa
├── train.py                       # Script huấn luyện tự động
└── evaluate.py                    # Script đánh giá kiểm thử độc lập
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt các thư viện cần thiết
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 2. Tiền xử lý dữ liệu và tạo đồ thị EDA
```bash
python src/data/run_preprocess.py
python src/util/eda.py
```

### 3. Huấn luyện các mô hình (Q-Learning vs Double Q-Learning)
```bash
python train.py
```

### 4. Đánh giá và so sánh trên tập kiểm thử chưa từng thấy (Unseen Test Data)
```bash
python evaluate.py
```

### 5. Khởi chạy ứng dụng Web trực quan Streamlit
```bash
streamlit run app/app.py
```

---

## 📈 Tóm Tắt Kết Quả Thực Nghiệm

| Thuật toán | Tổng Reward | Vi phạm Tiện nghi (%) | Điện năng Tiêu thụ (kWh) | Nhiệt độ TB Phòng (°C) |
| :--- | :---: | :---: | :---: | :---: |
| **Rule-Based (Thermostat)** | -93.44 | 56.42% | 459.84 | 25.03°C |
| **Q-Learning (Baseline)** | -92.31 | 68.00% | 423.88 | 25.32°C |
| **Double Q-Learning (Proposed)**| **-86.17** | **59.75%** | 606.32 | **24.12°C** |

- **Double Q-Learning** đạt **tổng Reward cao nhất (-86.17)** và duy trì nhiệt độ phòng ở mức tối ưu **24.12°C** (nằm hoàn hảo trong vùng tiện nghi ASHRAE $22^\circ\text{C} - 24.5^\circ\text{C}$).
- Cơ chế 2 bảng $Q_A$ và $Q_B$ giúp triệt tiêu **Maximization Bias (Ước lượng thừa)**, giúp bộ điều khiển đưa ra quyết định làm mát kịp thời khi phòng có dấu hiệu tăng nhiệt.
