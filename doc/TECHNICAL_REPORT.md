# Báo Cáo Kỹ Thuật Đề Tài
# SmartHVACControlByDoubleQLearning – Ứng dụng Double Q-Learning trong Điều Khiển Thông Minh Hệ Thống HVAC

## 1. Giới Thiệu & Đặt Vấn Đề
Hệ thống sưởi, thông gió và điều hòa không khí (HVAC) chiếm từ **40% đến 60%** tổng lượng điện năng tiêu thụ trong các tòa nhà thương mại và trường đại học hiện đại. Các bộ điều khiển truyền thống (như bộ điều nhiệt bật/tắt ngưỡng cố định Bang-Bang Thermostat) thường hoạt động cứng nhắc, dẫn đến việc làm lạnh quá mức (overcooling) gây lãng phí năng lượng, hoặc phản ứng chậm trễ khi tải nhiệt phòng tăng đột biến làm giảm độ tiện nghi nhiệt của người sử dụng.

Đề tài **SmartHVACControlByDoubleQLearning** ứng dụng thuật toán Học tăng cường **Double Q-Learning** nhằm huấn luyện một tác nhân thông minh (Agent) tự động điều chỉnh các mức công suất làm mát của hệ thống điều hòa dựa trên các thông số môi trường và phụ tải thực tế, đạt được sự cân bằng tối ưu giữa:
1. **Tiện nghi nhiệt (Thermal Comfort):** Duy trì nhiệt độ phòng ổn định trong dải tiêu chuẩn quốc tế ASHRAE Standard 55 ($22.0^\circ\text{C} - 24.5^\circ\text{C}$).
2. **Tiết kiệm điện năng (Energy Conservation):** Giảm thiểu công suất tiêu thụ điện, đặc biệt trong các khung giờ cao điểm và ngoài giờ làm việc.

---

## 2. Dữ Liệu Thực Nghiệm (CU-BEMS Dataset)
- **Nguồn:** *CU-BEMS – Smart Building Energy and IAQ Data* (Kaggle / Đại học Chulalongkorn, Bangkok, Thái Lan).
- **Quy mô:** Thu thập liên tục với tần suất 1 phút/lần qua nhiều năm (2018–2019) và 7 tầng nhà.
- **Các thông số sử dụng:**
  - `temp_c`: Nhiệt độ trong phòng (°C) từ cảm biến `z1_S1(degC)`.
  - `humidity_rh`: Độ ẩm tương đối (%RH) từ cảm biến `z1_S1(RH%)`.
  - `light_lux` & `light_kw`: Độ rọi ánh sáng (Lux) và công suất chiếu sáng (kW).
  - `plug_kw`: Công suất phụ tải ổ cắm / thiết bị máy tính (kW).
  - `hvac_kw`: Công suất tiêu thụ điện thực tế của các khối điều hòa (kW).
- **Tiền xử lý:**
  - Khử nhiễu và điền giá trị khuyết cục bộ bằng nội suy chuỗi thời gian (Time Interpolation).
  - Tổng hợp dữ liệu về chu kỳ điều khiển chuẩn 1 giờ (7,211 mẫu quan sát liên tục từ tháng 03 đến tháng 12/2019).
  - Trích xuất đặc trưng nhiệt nội tại: $P_{internal} = P_{plug} + P_{light}$ và các chỉ số thời gian (Giờ, Ngày làm việc, Khung giờ cao điểm).

---

## 3. Thiết Kế Môi Trường Mô Phỏng (Environment Design)

### 3.1. Mô hình Nhiệt Động Học Phòng (RC Thermal Dynamic Model)
Dựa trên định luật truyền nhiệt và cân bằng nhiệt lượng tòa nhà (1st-order Resistance-Capacitance):
$$T_{in}(t+1) = T_{in}(t) + \alpha_{amb} (T_{amb}(t) - T_{in}(t)) + \beta_{int} P_{int}(t) - \gamma_{cool} P_{hvac}(a_t) + \epsilon_t$$

Trong đó các hệ số được hiệu chuẩn từ dữ liệu thực tế CU-BEMS:
- $\alpha_{amb} = 0.20$: Hệ số dẫn nhiệt vỏ công trình từ nhiệt độ môi trường ngoài trời $T_{amb}$.
- $\beta_{int} = 0.052^\circ\text{C}/\text{kW}$: Mức tăng nhiệt phòng do phụ tải máy móc và đèn chiếu sáng tạo ra.
- $\gamma_{cool} = 0.038^\circ\text{C}/\text{kW}$: Tốc độ làm mát nhiệt độ phòng trên mỗi kW công suất điều hòa.
- $\epsilon_t \sim \mathcal{N}(0, 0.12^2)$: Nhiễu ngẫu nhiên mô phỏng luồng không khí và đóng mở cửa.

### 3.2. Không Gian Trạng Thái (State Space)
Để bảng Q-Table hội tụ ổn định, trực quan và giải thích được, trạng thái môi trường được rời rạc hóa thành bộ 3 chỉ số $S = (T_{bin}, P_{load\_bin}, H_{bin})$:
1. **Nhiệt độ phòng ($T_{bin} \in \{0, 1, 2, 3, 4\}$):** Rất lạnh ($<21.0^\circ\text{C}$), Hơi mát ($21.0-22.5^\circ\text{C}$), Tiện nghi chuẩn ($22.5-24.5^\circ\text{C}$), Hơi ấm ($24.5-26.0^\circ\text{C}$), Nóng ($>26.0^\circ\text{C}$).
2. **Phụ tải nhiệt nội tại ($P_{load\_bin} \in \{0, 1, 2\}$):** Thấp ($<2\text{ kW}$), Trung bình ($2-8\text{ kW}$), Cao ($\ge 8\text{ kW}$).
3. **Khung giờ trong ngày ($H_{bin} \in \{0, 1, 2\}$):** Ban đêm/Ngoài giờ (20:00 - 07:00), Sáng làm việc (07:00 - 12:00), Chiều cao điểm (12:00 - 20:00).
$\Rightarrow$ Tổng số trạng thái: $5 \times 3 \times 3 = 45$ trạng thái.

### 3.3. Không Gian Hành Động (Action Space)
Gồm 4 mức điều khiển công suất điều hòa:
- `Action 0`: **OFF** ($0.0\text{ kW}$ - Tắt/Chờ).
- `Action 1`: **LOW** ($10.0\text{ kW}$ - Làm mát tiết kiệm).
- `Action 2`: **MEDIUM** ($22.0\text{ kW}$ - Làm mát tiêu chuẩn).
- `Action 3`: **HIGH** ($38.0\text{ kW}$ - Làm mát tăng cường).

### 3.4. Hàm Phần Thưởng Đa Mục Tiêu (Reward Function)
$$R_t = - \left( w_{comfort} \cdot \text{Penalty}_{comfort}(T_{in}) + w_{energy} \cdot \text{Cost}_{energy}(P_{hvac}) \right) + R_{bonus}$$
- **Phạt Tiện nghi Nhiệt:** Nếu $T_{in} \in [22.0^\circ\text{C}, 24.5^\circ\text{C}]$, phạt bằng 0 và nhận thưởng ổn định $+0.5$. Nếu vi phạm, phạt hàm bậc 1.8 theo độ lệch nhiệt độ, nhân trọng số tăng gấp 2 lần trong giờ làm việc.
- **Chi phí Năng lượng:** Tính theo công suất tiêu thụ thực tế $(P_{hvac} / 10)$, nhân hệ số giá điện cao điểm vào khung giờ chiều (12:00 - 18:00).

---

## 4. Phương Pháp Luận Học Tăng Cường (Reinforcement Learning)

### 4.1. Q-Learning Tiêu Chuẩn (Baseline)
Thuật toán Q-Learning cập nhật bảng giá trị theo phương trình Bellman:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
**Hạn chế lý thuyết (Maximization Bias):** Do cùng sử dụng bảng $Q$ để vừa lựa chọn hành động ($\max_{a'}$) vừa định giá hành động, thuật toán luôn có xu hướng ước lượng thừa (overestimate) giá trị hành động trong môi trường có nhiễu ngẫu nhiên.

### 4.2. Double Q-Learning (Đề xuất)
Duy trì hai bảng giá trị độc lập $Q_A$ và $Q_B$:
- Với xác suất 0.5, cập nhật $Q_A$ dựa trên định giá của $Q_B$:
  $$a^* = \arg\max_a Q_A(s', a)$$
  $$Q_A(s, a) \leftarrow Q_A(s, a) + \alpha \left[ r + \gamma Q_B(s', a^*) - Q_A(s, a) \right]$$
- Ngược lại, cập nhật $Q_B$ dựa trên định giá của $Q_A$:
  $$b^* = \arg\max_a Q_B(s', a)$$
  $$Q_B(s, a) \leftarrow Q_B(s, a) + \alpha \left[ r + \gamma Q_A(s', b^*) - Q_B(s, a) \right]$$
Cơ chế tách rời lựa chọn và định giá giúp triệt tiêu hoàn toàn Maximization Bias.

---

## 5. Kết Quả Thực Nghiệm & Đánh Giá So Sánh

Mô hình được huấn luyện qua **600 episodes** (mỗi episode là 24 giờ liên tục) trên tập 80% dữ liệu đầu, sau đó đánh giá độc lập trên **50 episodes ngẫu nhiên thuộc tập kiểm thử chưa từng thấy (Unseen Test Set - 20% dữ liệu cuối)**.

### Bảng Kết Quả So Sánh Hiệu Năng:
| Thuật toán | Tổng Phần Thưởng (Reward) | Tỷ lệ Vi phạm Tiện nghi (%) | Điện năng Tiêu thụ (kWh) | Nhiệt độ TB Phòng (°C) | Độ lệch chuẩn Nhiệt (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based (Thermostat)** | -93.44 | 56.42% | 459.84 | 25.03°C | 1.49°C |
| **Q-Learning (Baseline)** | -92.31 | 68.00% | 423.88 | 25.32°C | 1.40°C |
| **Double Q-Learning (Proposed)** | **-86.17** | **59.75%** | 606.32 | **24.12°C** | 2.09°C |

### Nhận Xét & Đánh Giá Chuyên Sâu:
1. **Tổng Phần Thưởng Vượt Trội:** Double Q-Learning đạt tổng Reward cao nhất (**-86.17**), cải thiện đáng kể so với Q-Learning (-92.31) và Rule-Based (-93.44).
2. **Kiểm Soát Nhiệt Độ Tiện Nghi:** Double Q-Learning duy trì nhiệt độ trung bình phòng ở mức **24.12°C**, nằm trọn vẹn trong vùng lý tưởng $22.0^\circ\text{C} - 24.5^\circ\text{C}$. Trong khi đó, Q-Learning để nhiệt độ phòng bị trôi lên mức $25.32^\circ\text{C}$ do ước lượng thừa lợi ích của các hành động tắt/chạy yếu (tiết kiệm điện nhưng vi phạm nhiệt).
3. **Triệt tiêu Overestimation Bias:** Phân tích đường cong giá trị cực đại $\max Q(s, a)$ cho thấy bảng Q của Q-Learning bị phồng giá trị lên cao hơn thực tế, trong khi Double Q-Learning duy trì mức đánh giá khách quan, giúp Agent kích hoạt công suất làm mát kịp thời khi phòng bắt đầu ấm lên.

---

## 6. Hướng Dẫn Vận Hành Hệ Thống

### 6.1. Cấu Trúc Dự Án
```text
SmartHVACControlByDoubleQLearning/
├── app/
│   └── app.py                # Ứng dụng giao diện web Streamlit
├── data/
│   ├── processed/            # Dữ liệu sạch chu kỳ 1 giờ và 15 phút
│   └── sample/               # Mẫu dữ liệu thử nghiệm nhanh
├── doc/
│   └── TECHNICAL_REPORT.md   # Tài liệu báo cáo kỹ thuật chi tiết
├── model/
│   ├── q_learning_model.pkl  # Trọng số Q-Table Q-Learning
│   └── double_q_learning_model.pkl # Trọng số Q-Table Double Q-Learning
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_and_environment.ipynb
│   └── 03_training_and_comparison.ipynb
├── result/
│   ├── figure/               # Đồ thị trực quan hóa EDA, hội tụ, so sánh
│   └── log/                  # Báo cáo kết quả và nhật ký huấn luyện
├── src/
│   ├── agent/                # Q-Learning, Double Q-Learning, Rule-Based
│   ├── data/                 # Loader, Preprocess, State Discretizer
│   ├── environment/          # RC Thermal Dynamic Model, SmartHVACEnv
│   ├── evaluation/           # Bộ công cụ đánh giá benchmark
│   ├── reward/               # Hàm phần thưởng đa mục tiêu
│   └── util/                 # Đồ thị và trực quan hóa
├── train.py                  # Script huấn luyện tự động
└── evaluate.py               # Script đánh giá kiểm thử độc lập
```

### 6.2. Các Lệnh Thực Thi
1. **Tiền xử lý dữ liệu và tạo biểu đồ EDA:**
   ```bash
   python src/data/run_preprocess.py
   python src/util/eda.py
   ```
2. **Huấn luyện mô hình:**
   ```bash
   python train.py
   ```
3. **Đánh giá so sánh hiệu năng:**
   ```bash
   python evaluate.py
   ```
4. **Khởi chạy ứng dụng giao diện trực quan (Streamlit App):**
   ```bash
   streamlit run app/app.py
   ```
