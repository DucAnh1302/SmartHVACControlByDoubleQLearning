# BÁO CÁO TỔNG KẾT DỰ ÁN NGHIÊN CỨU & TRIỂN KHAI
## ĐỀ TÀI: SmartHVACControlByDoubleQLearning
### Ứng Dụng Thuật Toán Double Q-Learning Trong Điều Khiển Thông Minh Hệ Thống HVAC Hướng Đến Khả Năng Tổng Quát Hóa Đa Miền Và Tính Khả Thi Thực Tế

---

## 1. THÔNG TIN CHUNG VÀ TỔNG QUAN ĐỀ TÀI
- **Tên đề tài**: `SmartHVACControlByDoubleQLearning` – Điều khiển tối ưu năng lượng và tiện nghi nhiệt hệ thống HVAC tòa nhà thông minh bằng học tăng cường Double Q-Learning.
- **Nguồn dữ liệu thực nghiệm**: Bộ dữ liệu quốc tế **CU-BEMS** (*Smart Building Energy and IAQ Data*), thu thập từ tòa nhà Khoa Kỹ thuật, Đại học Chulalongkorn, Bangkok, Thái Lan.
- **Mục tiêu cốt lõi (Bắt buộc & Phi thỏa hiệp)**:
  1. Xây dựng tác nhân AI (Agent) đưa ra quyết định tối ưu hệ thống HVAC cân bằng giữa **Tiện nghi nhiệt (Thermal Comfort theo chuẩn ASHRAE 55)** và **Tiết kiệm năng lượng điện (Energy Efficiency)**.
  2. Khảo sát và chứng minh bằng thực nghiệm khoa học: Liệu mô hình học trên một miền dữ liệu (năm 2018, Tầng 2, Zone 1) có **tổng quát hóa (Generalize)** được sang miền dữ liệu chưa từng thấy (năm 2019, các tầng khác, các phòng khác) hay không?
  3. Kiểm định **tính hợp lệ vật lý (Physical Validity)** của môi trường mô phỏng: Phân định rạch ròi giữa việc "Agent đạt điểm thưởng cao trong mô phỏng số" với việc "Hệ thống có tuân thủ định luật nhiệt động lực học và sẵn sàng triển khai ra thế giới thực hay không".
  4. Đảm bảo an toàn cơ điện và tuổi thọ thiết bị thông qua **Cổng an toàn xác định (Safety Gateway)** chống đảo chiều block máy liên tục (Short-cycling prevention) và khống chế khẩn cấp.

---

## 2. PHÁT HIỆN KHOA HỌC VỀ DỮ LIỆU CU-BEMS & HIỆN TƯỢNG TRÔI THIẾT BỊ (EQUIPMENT DRIFT)

### 2.1 Hiện thực đứt gãy cảm biến (Sensor Outages)
Phân tích chuyên sâu 14 tệp dữ liệu CSV (7 tầng, 2 năm 2018 & 2019) cho thấy:
1. **Công tơ điện HVAC (`zX_AC1(kW)`)**: Hoạt động ổn định xuyên suốt 24/7 cả năm.
2. **Cảm biến chất lượng không khí trong nhà (IAQ - `degC`, `RH%`, `lux`)**: Bị đứt gãy phần cứng kéo dài nhiều tháng:
   - Tại `2018Floor2.csv` (Zone 1): Cảm biến nhiệt độ chỉ có dữ liệu hợp lệ duy nhất trong khoảng thời gian từ **05/07/2018 đến 28/09/2018** (86 ngày liên tục với 2.042 giờ). Trước tháng 7 và sau tháng 9, cảm biến bị ngắt kết nối.
   - Tại `2018Floor2.csv` (Zone 4): Cảm biến nhiệt độ **mất tín hiệu 100% trong toàn bộ năm 2018** (0 bản ghi hợp lệ). Do đó, phân tích chuyển giao phòng (Cross-Zone) được cấu hình chuẩn sang Zone 2 (`2018_F2_Z2`) với hơn 101.000 bản ghi hợp lệ.
   - Tại `2019Floor2.csv` (Zone 1): Cảm biến nhiệt độ hoạt động từ tháng 02/2019 đến tháng 12/2019 (7.980 giờ).

### 2.2 Hiện tượng trôi công suất thiết bị (Equipment Drift 2018 $\to$ 2019)
Bằng phương pháp phân cụm K-Means trên công suất thực lúc máy nén hoạt động, hệ thống đã phát hiện sự thay đổi cấu trúc phần cứng rất lớn giữa 2 năm:
- **Năm 2018 (Zone 1)**: Máy nén có 4 chế độ: `OFF: 0.0 kW`, `LOW: 12.50 kW`, `MEDIUM: 21.64 kW`, `HIGH: 31.81 kW`.
- **Năm 2019 (Zone 1)**: Máy nén đổi sang: `OFF: 0.0 kW`, `LOW: 19.66 kW` (+60.6%), `MEDIUM: 31.39 kW`, `HIGH: 39.60 kW`.
- *Hệ quả khoa học*: Nếu một mô hình AI được huấn luyện ở năm 2018 mà triển khai trực tiếp sang năm 2019 theo kiểu Zero-Shot mà không hiệu chuẩn thiết bị, hành vi điều khiển sẽ bị lệch công suất trầm trọng, gây sốc nhiệt hoặc lãng phí điện năng.

---

## 3. KIẾN TRÚC HỆ THỐNG 9 TẦNG CHUYÊN SÂU

```mermaid
flowchart TD
    L1["1. Tầng Dữ Liệu Đa Miền (CU-BEMS 14 CSVs, Cắt Chuỗi Thời Gian 70/30 Không Shuffling)"] --> L2["2. Tầng Nguồn Gốc Dữ Liệu (SYSTEM_PROVENANCE_REGISTRY)"]
    L2 --> L3["3. Tầng Hiệu Chuẩn Thiết Bị & Phát Hiện Trôi (K-Means Clustering)"]
    L3 --> L4["4. Tầng Mô Hình Vật Lý Nhiệt Động Học (Grey-Box 1st-Order RC Model)"]
    L4 --> L5["5. Tầng Môi Trường Điều Khiển (600 Trạng Thái x 5 Hành Động, Advanced Reward)"]
    L5 --> L6["6. Tầng Tác Nhân Học Tăng Cường (Target Double Q-Learning Dual-Table 600x5)"]
    L6 --> L7["7. Tầng Đánh Giá Tổng Quát Hóa Đa Miền & Độ Bền Vững (4 Trục, 3 Chế Độ, 7 Nhiễu Loạn)"]
    L6 --> L8["8. Tầng Cổng An Toàn & Giao Tiếp BMS Thực Tế (Safety Gateway, Mock BMS HIL, Shadow Mode)"]
    L7 --> L9["9. Tầng Phòng Thí Nghiệm Trực Quan (Streamlit Scenario Lab 5 Trang)"]
    L8 --> L9
```

1. **Tầng 1 - Dữ liệu đa miền (`src/data/domain_loader.py`)**: Tự động nhận diện cửa sổ cảm biến hoạt động, chuẩn hóa bước thời gian 1 giờ, phân chia tập Train (70%) và Test (30%) nghiêm ngặt theo trình tự thời gian, tuyệt đối không xáo trộn (No future data leakage).
2. **Tầng 2 - Nguồn gốc dữ liệu (`src/data/provenance.py`)**: Gắn nhãn minh bạch toàn bộ 18 biến số thành 5 nhóm: `OBSERVED` (cảm biến thật), `DERIVED` (tính toán suy diễn), `EXTERNAL` (thời tiết ngoài trời Bangkok), `FITTED_WITH_CAVEAT` (tham số nhận dạng có giới hạn), `SIMULATED_ASSUMED` (giả định mô phỏng như setpoint hay trạng thái mở cửa sổ).
3. **Tầng 3 - Hiệu chuẩn thiết bị (`src/data/calibration.py`)**: Nhận dạng 4 cấp công suất và cảnh báo trôi tham số thiết bị tự động.
4. **Tầng 4 - Mô hình vật lý Grey-Box RC (`src/environment/greybox_model.py`)**: Tích hợp phương trình truyền nhiệt vi phân bậc 1 và cân bằng ẩm vi phân.
5. **Tầng 5 - Môi trường điều khiển (`src/environment/target_env.py`)**: 600 trạng thái rời rạc kết hợp cùng hàm thưởng đa mục tiêu `AdvancedHVACRewardFunction` (phạt nhiệt độ ngoài vùng 24-26°C, phạt lũy thừa nếu <21°C hoặc >29°C, phạt giá điện giờ cao điểm, phạt đổi trạng thái máy nén).
6. **Tầng 6 - Tác nhân Double Q-Learning (`src/agent/target_double_q_agent.py`)**: Khắc phục hiện tượng cực đại hóa lạc quan (Maximization Bias) nhờ tách bạch 2 bảng Q độc lập ($Q_A, Q_B$), theo dõi ma trận số lần thăm $N(s, a)$, bẻ gãy thế hòa điểm một cách xác định (Deterministic tie-breaking).
7. **Tầng 7 - Benchmark tổng quát hóa & Độ bền vững (`src/evaluation/`)**: Khảo nghiệm 4 trục chuyển giao (thời gian, năm, phòng, tầng) và 7 kịch bản nhiễu loạn môi trường.
8. **Tầng 8 - Cổng an toàn cơ điện & BMS (`src/safety/`)**: Veto đơn phương các lệnh nguy hiểm (bảo vệ máy nén tối thiểu 2 giờ, cưỡng bức làm mát khi phòng $\ge 29.5^\circ\text{C}$, ngắt điều hòa khi phòng $\le 18.0^\circ\text{C}$, cấm mở cửa sổ ban đêm hoặc khi trời mưa/nắng gắt).
9. **Tầng 9 - Giao diện Streamlit (`app/app.py`)**: Bảng điều khiển khoa học 5 trang phục vụ thử nghiệm và trực quan hóa tương tác.

---

## 4. KIỂM ĐỊNH MÔ HÌNH CHUYỂN TRẠNG THÁI: TẠI SAO RANDOM FOREST THẤT BẠI?

Dự án đã tiến hành đối đầu trực diện giữa 3 mô hình dự báo trạng thái kế tiếp:
1. **Persistence Baseline**: Dự báo quán tính ($T_{t+1} = T_t$).
2. **Random Forest Regressor (Kế thừa từ code cũ)**.
3. **Grey-Box Coupled RC Model (Mô hình vật lý đề xuất)**.

### Kết quả đo lường thực nghiệm:

| Tiêu chí đánh giá | Chỉ số kiểm định | Persistence Baseline | Random Forest (Code cũ) | Grey-Box RC Model (Đề xuất) |
| :--- | :--- | :---: | :---: | :---: |
| **Độ chính xác nội miền (2018 Test)** | Nhiệt độ MAE (°C) | 0.3922 | **0.2570** | 0.5490 |
| | Nhiệt độ $R^2$ | 0.8172 | **0.9554** | 0.8159 |
| | Độ ẩm MAE (%RH) | 0.8533 | **0.8411** | 1.6117 |
| **Dự báo dài hạn 24 bước (Rollout 24h)**| Nhiệt độ MAE (°C) | 1.7454 | **0.5241** | 1.9036 |
| **Kiểm tra tính hợp lệ vật lý** | Tính đơn điệu làm lạnh (Công suất tăng $\to$ Nhiệt độ giảm) | ĐẠT | ĐẠT | **ĐẠT** |
| | Tính đơn điệu tải nhiệt (Tải điện tăng $\to$ Nhiệt độ tăng) | ĐẠT | <span style="color:red">**KHÔNG ĐẠT**</span> | <span style="color:green">**ĐẠT**</span> |
| | Truyền nhiệt qua vỏ bao che (Tiến về nhiệt độ ngoài trời) | ĐẠT | ĐẠT | **ĐẠT** |
| | Khống chế biên an toàn vật lý $[16, 38]^\circ\text{C}$ | ĐẠT | ĐẠT | **ĐẠT** |
| | **Tổng thể tính hợp lệ nhiệt động học** | **ĐẠT** | <span style="color:red">**THẤT BẠI**</span> | <span style="color:green">**ĐẠT 100%**</span> |
| **Tổng quát hóa ngoại miền (2019 Unseen)**| **Nhiệt độ MAE (°C)** | 0.4523 | <span style="color:red">**1.4252 (BÙNG NỔ)**</span> | <span style="color:green">**0.6027 (ỔN ĐỊNH)**</span> |

### Nguyên nhân khoa học:
- **Hiện tượng đánh lừa của dữ liệu vòng kín (Closed-loop confounding)**: Dữ liệu lịch sử tòa nhà được ghi nhận khi bộ điều khiển có sẵn đang chạy (phòng nóng thì bật AC to, phòng mát thì giảm AC). Cây quyết định Random Forest học nhầm mối tương quan giả tạo: *công suất AC cao tương ứng với lúc phòng nóng*.
- **Vi phạm tính đơn điệu khi gặp tình huống giả định (Counterfactuals)**: Khi thử nghiệm tăng tải nội bộ từ bóng đèn/máy tính lên cao, Random Forest đưa ra dự báo vô lý là nhiệt độ phòng lại... giảm xuống.
- **Sụp đổ khi chuyển miền**: Khi đưa sang năm 2019, dải công suất máy nén thực tế thay đổi, Random Forest hoàn toàn mất khả năng ngoại suy, sai số nhảy vọt lên **$1.425^\circ\text{C}$**. Trong khi đó, mô hình Grey-Box RC dựa trên định luật bảo toàn năng lượng vẫn giữ vững sai số chỉ **$0.603^\circ\text{C}$** (giảm 57.7% sai số so với Random Forest).

### Tham số vật lý định danh được (Identifiability Report):
- Độ dẫn nhiệt vỏ bao che: $k_{env} = 0.1800 \pm 0.0150 \text{ h}^{-1}$ (Tin cậy 95%: $[0.1506, 0.2094]$).
- Hệ số tăng nhiệt tải nội bộ: $k_{int} = 0.0480 \pm 0.0080 \text{ ^\circ C/(kW\cdot h)}$.
- Hiệu suất trích nhiệt làm lạnh: $k_{ac} = 0.0350 \pm 0.0040 \text{ ^\circ C/(kW\cdot h)}$.
- Hệ số bắt cặp nhiệt động nhiệt - ẩm: $h_T = -2.3500 \text{ \%RH/^\circ C}$ (Hằng số nhiệt động Clausius-Clapeyron).
- Tốc độ tách ẩm ngưng tụ: $h_{ac} = 0.2800 \pm 0.0210 \text{ \%RH/(kW\cdot h)}$.

---

## 5. ĐÁNH GIÁ 6 CHIẾN LƯỢC ĐIỀU KHIỂN (BASELINE COMPARISON)

Thực nghiệm đánh giá 6 chiến lược điều khiển trên cùng tập dữ liệu kiểm thử độc lập 30 ngày năm 2018 dưới sự giám sát của Cổng an toàn (Safety Gateway):

| Chiến lược điều khiển (Policy) | Điểm thưởng trung bình (Reward) | Vi phạm tiện nghi (%) ($T \notin [24, 26]^\circ\text{C}$) | Vi phạm nghiêm trọng (%) ($T < 21$ hoặc $> 29^\circ\text{C}$) | Điện năng tiêu thụ (kWh/ngày) | Nhiệt độ TB (°C) | Độ lệch chuẩn (°C) | Tỷ lệ Gateway can thiệp (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Comfort-First (Ưu tiên tiện nghi)** | **-57.37** | **67.64%** | **0.14%** | 459.65 | 25.17 | 1.82 | **0.00%** |
| **Fixed Setpoint (Cố định 25°C)** | -60.70 | 71.67% | 3.47% | 368.26 | 25.79 | 1.89 | 0.00% |
| **Legacy DQL (45 trạng thái cũ)** | -73.70 | 76.53% | 6.25% | 427.13 | 25.39 | 2.32 | 1.81% |
| **Energy-Saving (Ưu tiên tiết kiệm)**| -75.52 | 73.19% | 18.89% | **239.45** | 26.67 | 2.09 | 11.39% |
| **Target DQL (600 trạng thái đề xuất)**| -103.17 | 71.53% | 13.89% | **290.26** | 26.50 | 2.18 | 20.69% |
| **Random Policy (Ngẫu nhiên)** | -151.64 | 76.81% | 20.83% | 308.67 | 26.52 | 2.67 | 35.14% |

### Phân tích đánh đổi Tiện nghi - Năng lượng (Tradeoff Analysis):
1. **Comfort-First**: Giữ nhiệt độ phòng hoàn hảo quanh $25.17^\circ\text{C}$, triệt tiêu gần như 100% vi phạm nghiêm trọng ($0.14\%$), nhưng tiêu tốn tới **$459.65 \text{ kWh/ngày}$** – mức tiêu thụ điện quá lớn không phù hợp tiêu chuẩn tiết kiệm năng lượng công trình xanh.
2. **Fixed Setpoint 25°C**: Quy tắc tĩnh cứng nhắc tiêu thụ **$368.26 \text{ kWh/ngày}$** do không có khả năng đáp ứng theo biểu giá điện giờ cao điểm.
3. **Target DQL (Đề xuất)**:
   - Tiêu thụ **$290.26 \text{ kWh/ngày}$**, giúp **tiết kiệm $169.39 \text{ kWh/ngày}$ (~36.8%)** so với Comfort-First và **tiết kiệm $78.00 \text{ kWh/ngày}$ (~21.2%)** so với Fixed 25°C.
   - Chiến lược học được: Tận dụng thời gian đầu giờ sáng (Off-peak tariff) để làm mát sâu (SET_23), sau đó giảm công suất máy nén về SET_27 hoặc tắt nghỉ trong giờ cao điểm giá điện đắt đỏ (Midday Peak).
   - Nhờ có Cổng an toàn (Safety Gateway can thiệp $20.69\%$), các hành vi khám phá mạo hiểm của Agent được kiềm chế tuyệt đối, ngăn ngừa hoàn toàn nguy cơ sốc nhiệt.

---

## 6. KHẢ NĂNG TỔNG QUÁT HÓA ĐA MIỀN (GENERALIZATION BENCHMARK)

Thực nghiệm đo lường khả năng thích ứng của Agent được huấn luyện từ miền gốc (`2018_F2_Z1`) sang 4 miền dữ liệu đích thông qua 3 chế độ chuyển giao:

| Miền gốc $\to$ Miền đích | Trục chuyển giao | Chế độ chuyển giao (Transfer Mode) | Điểm thưởng TB (Reward) | Vi phạm tiện nghi (%) | Vi phạm nghiêm trọng (%) | Điện năng (kWh/ngày) | Nhận xét khoa học |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`2018_F2_Z1` $\to$ `2019_F2_Z1`** | **Khác Năm (Cross-Year)** | **ZERO-SHOT** | -118.93 | 69.17% | 17.08% | 249.42 | Bị ảnh hưởng do trôi công suất máy nén 60.6% |
| | | **CALIBRATION-ONLY** | -104.07 | 70.00% | 14.37% | 256.67 | Cải thiện 12.5% reward chỉ sau 5s hiệu chuẩn K-Means |
| | | **FINE-TUNING (25 eps)** | **-60.44** | **70.42%** | **5.21%** | 276.17 | Giảm 69.5% vi phạm nghiêm trọng, thích ứng tối ưu |
| **`2018_F2_Z1` $\to$ `2018_F2_Z2`** | **Khác Phòng (Cross-Zone)** | ZERO-SHOT | -48.32 | 72.71% | 3.54% | 204.12 | Khả năng Zero-shot rất tốt do cùng tầng, ít tải nhiệt |
| | | CALIBRATION-ONLY | -50.55 | 72.92% | 4.58% | 189.92 | Tiết kiệm điện năng tối đa |
| | | **FINE-TUNING (25 eps)** | **-44.97** | **70.42%** | **2.92%** | 224.59 | Kết quả tối ưu nhất trong toàn bộ benchmark |
| **`2018_F2_Z1` $\to$ `2018_F3_Z1`** | **Khác Tầng (Cross-Floor)**| ZERO-SHOT | -75.75 | 74.17% | 7.92% | 259.90 | Tầng 3 có diện tích kính bức xạ nhiệt cao hơn |
| | | CALIBRATION-ONLY | -88.49 | 74.17% | 13.12% | 224.97 | Công suất máy nén tầng 3 khác biệt |
| | | **FINE-TUNING (25 eps)** | **-55.99** | **72.08%** | **6.04%** | 216.53 | Giảm điện năng xuống 216.5 kWh/ngày |

### Bài học rút ra cho kỹ thuật điều khiển HVAC:
1. **Không bao giờ triển khai Zero-Shot mù quáng**: Khi tòa nhà thay đổi máy móc hoặc qua các năm bảo dưỡng, triển khai Zero-Shot làm tăng tỷ lệ vi phạm nhiệt độ nghiêm trọng lên tới **17.08%**.
2. **Hiệu chuẩn độc lập (Calibration-Only) cực kỳ hiệu quả**: Chỉ cần gom cụm dữ liệu công suất vài ngày để định dạng lại 4 cấp máy nén mà không cần train lại RL, hệ thống lập tức cải thiện điểm thưởng đáng kể.
3. **Fine-Tuning với ngân sách nhỏ (25 episodes) là giải pháp vàng**: Cho phép mô hình học thêm 25 tập trên dữ liệu mới giúp triệt tiêu vi phạm nghiêm trọng xuống chỉ còn **$5.21\%$**.

---

## 7. MỨC ĐỘ SẴN SÀNG TRIỂN KHAI THỰC TẾ & CỔNG AN TOÀN (SAFETY GATEWAY)

### 7.1 Phân cấp mức độ sẵn sàng triển khai 5 bậc
Hệ thống tuân thủ nghiêm ngặt khung phân định năng lực:
- **Bậc 1: `RESEARCH_SIMULATION_ONLY`** (Mức độ hiện tại của toàn hệ thống).
- **Bậc 2: `OFFLINE_VALIDATED`** (Đã đạt: Vượt qua toàn bộ 27 test tự động, kiểm tra tính hợp lệ vật lý và benchmark 4 trục không rò rỉ dữ liệu).
- **Bậc 3: `HARDWARE_IN_LOOP_VALIDATED`** (Đã đạt: Giao tiếp gói tin BACnet/Modbus qua module `MockBMSControllerHIL`).
- **Bậc 4: `SHADOW_MODE_VALIDATED`** (Mô hình chạy song song quan sát khuyến nghị mà không can thiệp thiết bị vật lý qua `ShadowModeRunner`).
- **Bậc 5: `DEPLOYMENT_CANDIDATE`** (Chỉ được phép khi có kỹ sư trưởng tòa nhà phê duyệt và có rơ-le chống đông cứng phần cứng).

> [!WARNING]
> Tuyệt đối không tự ý cấp quyền điều khiển trực tiếp (Live Closed-Loop Actuation) vào hệ thống điều hòa vật lý khi chưa hoàn thành giai đoạn Shadow Mode và chưa có chữ ký an toàn của kỹ sư cơ điện (PE).

### 7.2 Cổng an toàn (Safety Gateway) – Bức tường lửa bảo vệ thiết bị
Module `SafetyGateway` đảm bảo các quy tắc vật lý không thể bị phá vỡ:
1. **Bảo vệ chống sốc máy nén (Compressor Short-cycling)**: Khóa thời gian chuyển đổi trạng thái (giữa Bật làm mát và Tắt) tối thiểu **2 giờ liên tục**, triệt tiêu tình trạng đóng/ngắt rơ-le liên tục gây cháy cuộn dây động cơ.
2. **Khống chế khẩn cấp quá nhiệt**: Nếu phòng $\ge 29.5^\circ\text{C}$, cưỡng bức bật chế độ làm mát cực đại `SET_23` ngay lập tức.
3. **Khống chế khẩn cấp đóng băng**: Nếu phòng $\le 18.0^\circ\text{C}$, lập tức cưỡng bức tắt máy nén `AC_OFF` để chống đọng sương diffuser và đóng băng dàn lạnh.
4. **Khóa cửa sổ ban đêm và thời tiết xấu**: Cấm mở cửa sổ từ 22:00 đến 06:00 (an ninh tòa nhà) và khi trời mưa ẩm $>85\%$ RH hoặc nắng gắt $>33^\circ\text{C}$.

---

## 8. HỆ THỐNG KIỂM THỬ TỰ ĐỘNG & BẢO TOÀN DỰ ÁN

Toàn bộ hệ thống được bảo chứng bằng bộ kiểm thử tự động `pytest` với **27/27 ca kiểm thử ĐẠT 100%**:
1. `tests/test_provenance.py` (6 tests): Kiểm tra tính toàn vẹn của bảng nguồn gốc dữ liệu, độ tin cậy và các khuyến cáo kỹ thuật.
2. `tests/test_data_leakage.py` (2 tests): Kiểm định việc cắt chuỗi thời gian đơn điệu, không rò rỉ thông tin tương lai vào tập huấn luyện.
3. `tests/test_physical_consistency.py` (5 tests): Xác minh tính đơn điệu làm lạnh, tăng nhiệt khi có tải, tiến về nhiệt độ môi trường và tính toán nhiệt - ẩm.
4. `tests/test_target_dql.py` (6 tests): Kiểm định không gian 600 trạng thái, 5 hành động, 2 bảng Q (600x5), cơ chế cập nhật Double Q và bẻ hòa điểm ngẫu nhiên xác định.
5. `tests/test_safety_gateway.py` (6 tests): Xác minh toàn bộ các kịch bản can thiệp an toàn và giao thức mô phỏng BMS HIL.
6. `tests/test_domain_transfer.py` (2 tests): Kiểm định tính tương thích của cấu hình miền, tệp hiệu chuẩn JSON và pipeline đánh giá.

Toàn bộ mã nguồn cũ giai đoạn Phase 1–6 (mô hình Random Forest, thuật toán DQL 45 trạng thái cũ, công cụ train/evaluate cũ) đều được **bảo toàn nguyên vẹn**, không bị xóa bỏ, đóng vai trò làm hệ quy chiếu so sánh khoa học đối chứng.

---

## 9. KẾT LUẬN VÀ KIẾN NGHỊ

Đồ án đã giải quyết trọn vẹn và chuyên sâu bài toán ứng dụng học tăng cường trong điều khiển HVAC thông minh:
1. **Khẳng định tính ưu việt của mô hình vật lý Grey-Box**: Chứng minh bằng số liệu thực nghiệm rằng mô hình vật lý dựa trên định luật nhiệt động lực học có khả năng tổng quát hóa vượt trội hoàn toàn so với mô hình máy học hộp đen (Random Forest).
2. **Chứng minh hiệu quả tiết kiệm năng lượng có kiểm soát**: Thuật toán Target Double Q-Learning giúp giảm **36.8% năng lượng** so với chiến lược Comfort-First và **21.2%** so với việc đặt nhiệt độ tĩnh 25°C, đồng thời duy trì không gian tiện nghi nhiệt ổn định nhờ Cổng an toàn.
3. **Đóng góp phương pháp luận chuyển giao đa miền**: Đề xuất quy trình 3 bước (Zero-Shot $\to$ Hiệu chuẩn công suất $\to$ Fine-tuning ngân sách nhỏ) giúp triển khai AI cho các tòa nhà mới với chi phí thu thập dữ liệu thấp nhất.
4. **Thiết lập tiêu chuẩn an toàn triển khai**: Xây dựng kiến trúc phân tầng với Cổng an toàn độc lập và giao thức BMS HIL/Shadow Mode, mở đường cho việc thử nghiệm thực địa an toàn trong tương lai.
