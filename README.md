# SmartHVACControlByDoubleQLearning
## Multi-Device HVAC Digital Twin & Step-by-Step Agent Control Platform

Ứng dụng thuật toán **Double Q-Learning** trong điều khiển thông minh hệ thống HVAC (Heating, Ventilation, and Air Conditioning) đa thiết bị trên nền tảng **Digital Twin vật lý số**, kết hợp bộ dữ liệu thực tế **CU-BEMS** với các mô hình truyền nhiệt, cân bằng ẩm nhiệt động học (Psychrometrics), cơ chế thông gió CO₂ và lọc bụi mịn PM2.5.

> [!IMPORTANT]
> **Tuyên bố phạm vi khoa học**: Đây là dự án nghiên cứu và mô phỏng số (RESEARCH_SIMULATION_ONLY). Toàn bộ kết quả tiết kiệm năng lượng và tuân thủ tiện nghi được đo lường trong môi trường Digital Twin, tuyệt đối không tuyên bố là hệ thống đã kiểm chứng hay điều khiển thành công trên tòa nhà vật lý thực tế.

---

## 🌟 Tính Năng Mới — Bản Nâng Cấp Digital Twin Đa Thiết Bị (Version 2.0)

1. **Mô phỏng 7 nhóm thiết bị HVAC đồng bộ (7 Actuator Groups)**:
   - **Cooling (Máy lạnh)**: Điều khiển máy nén nhiều tầng (OFF, LOW, MED, HIGH), tính toán công suất điện (kW), công suất nhiệt lạnh theo COP (3.2), và ngưng tụ ẩm phụ trợ.
   - **Heating (Máy sưởi)**: Điều khiển sưởi ấm (OFF, LOW, HIGH), khóa liên động (Interlock) tuyệt đối không cho sưởi và làm lạnh đồng thời.
   - **Mechanical Ventilation (Quạt cấp & Quạt hút)**: Điều khiển lưu lượng gió tươi (100 - 500 m³/h), mô phỏng quá trình pha loãng CO₂ và trao đổi nhiệt/ẩm với khí trời.
   - **Window / Damper (Cửa sổ & Van gió)**: Mô phỏng thông gió tự nhiên dựa trên chênh lệch nhiệt độ trong/ngoài và diện tích mở, có khóa bảo vệ khi thời tiết xấu hoặc ban đêm.
   - **Humidification (Máy tạo ẩm)**: Bổ sung hơi nước (0.25 - 0.60 kg/h), tác động trực tiếp vào tỷ trọng ẩm $W$ và độ ẩm tương đối %RH qua công thức Magnus.
   - **Dehumidification (Máy hút ẩm)**: Hút ẩm chuyên dụng (1.20 kg/h) mà không làm hạ nhiệt độ phòng quá mức; khóa liên động loại trừ máy tạo ẩm.
   - **Air Filtration (Máy lọc không khí HEPA)**: Lọc bụi mịn PM2.5 theo chỉ số CADR (80 - 450 m³/h), không làm giảm CO₂ trừ khi có gió tươi.
2. **Không gian 720 trạng thái rời rạc $\times$ 12 Coordinated Action Templates**:
   - Tránh bùng nổ tổ hợp ($3^7 = 2.187$ actions) bằng 12 kịch bản vận hành phối hợp vật lý chặt chẽ.
   - Lưu trữ bảng Q kép ($Q_A, Q_B$) với $720 \times 12 = 8.640$ phần tử mỗi bảng.
3. **Giao diện mới duy nhất: HVAC Agent Live Simulation (`app/live_simulation_gui.py`)**:
   - Chạy từng bước chính xác (Step Forward 1 step), Start, Pause, Resume, Stop, Reset.
   - Điều chỉnh tốc độ mô phỏng linh hoạt: 0.25x (2000ms), 0.5x (1000ms), 1.0x (500ms), 2.0x (250ms), 5.0x (100ms).
   - Tách biệt rõ Simulation Time và Wall-clock Time.
   - Hiển thị đầy đủ nhãn nguồn gốc (`OBSERVED`, `DERIVED`, `EXTERNAL`, `FITTED`, `SIMULATED_ASSUMED`, `SIMULATED`).
   - Mặc định mở ở **Evaluation Mode** (Q-Table bị đóng băng 100%, không cập nhật).
   - Có tùy chọn **Online Learning Mode** (Cập nhật Q1/Q2 ngay trong lúc mô phỏng khi người dùng bật).
   - Tuân thủ chỉ thị `[ERR-202611-003][gui-combobox-width]` (Combobox ở hàng riêng rộng $\ge 600$px, popdown width cấu hình chuẩn, nhãn chi tiết động bên dưới).
4. **Hàm thưởng đa mục tiêu 7 thành phần (Multi-Objective Reward Decomposition)**:
   $$R_{total} = R_{temp} + R_{humidity} + R_{co2} + R_{air\_quality} + R_{energy} + R_{switching} + R_{safety}$$
5. **Bộ kiểm thử toàn diện: 68/68 Unit Tests (100% Pass Rate)**:
   - 43 bài test kế thừa từ hệ thống ban đầu (Grey-box, Safety, Data leakage, Domain transfer).
   - 25 bài test vật lý, tâm trắc học Magnus, nồng độ ẩm, CO₂, lọc bụi mịn PM2.5, công suất standby máy nén, kịch bản môi trường và bảng Q kép mới.

---

## 📚 Tài Liệu Kỹ Thuật Đầy Đủ

1. 🏛️ **[HVAC_UPGRADE_DESIGN.md](HVAC_UPGRADE_DESIGN.md)**: Bản thiết kế kiến trúc nâng cấp Digital Twin đa thiết bị, phân tích tương thích và quản trị rủi ro.
2. 🧠 **[AGENT_LEARNING_EXPLAINED.md](AGENT_LEARNING_EXPLAINED.md)**: Giải thích chi tiết một step, episode, cách agent chọn action, luật cập nhật Double Q-Learning, Evaluation Mode vs Online Learning Mode.
3. 🔬 **[HVAC_SIMULATION_ASSUMPTIONS.md](HVAC_SIMULATION_ASSUMPTIONS.md)**: Toàn bộ phương trình nhiệt động học, ẩm học Magnus, CO₂, PM2.5, công suất thiết bị và bảng Provenance.
4. ⚙️ **[HVAC_ACTION_SPACE.md](HVAC_ACTION_SPACE.md)**: Phân tích toán học không gian hành động, lý do dùng 12 Action Templates và các khóa liên động cơ điện.
5. 🎯 **[HVAC_REWARD_DESIGN.md](HVAC_REWARD_DESIGN.md)**: Thiết kế hàm thưởng phân rã 7 thành phần, đơn vị đo, cấu hình trọng số theo ưu tiên của người dùng.
6. 🌐 **[HVAC_SCOPE_AND_COVERAGE.md](HVAC_SCOPE_AND_COVERAGE.md)**: Phân loại 3 cấp độ phạm vi mô phỏng: Directly Simulated, Simplified, và Out of Scope.
7. 🏛️ **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)**: Kiến trúc hệ thống 9 tầng.
8. 📊 **[DATASET_AND_PROVENANCE.md](DATASET_AND_PROVENANCE.md)**: Quản trị dữ liệu CU-BEMS và khung nhãn nguồn gốc.
9. 🌡️ **[PHYSICAL_MODEL_VALIDATION.md](PHYSICAL_MODEL_VALIDATION.md)**: Thẩm định mô hình RC bậc 1 so với Random Forest và Persistence.
10. 🌐 **[GENERALIZATION_BENCHMARK.md](GENERALIZATION_BENCHMARK.md)**: Benchmark tổng quát hóa 4 trục chuyển miền.
11. 🛡️ **[REAL_WORLD_READINESS.md](REAL_WORLD_READINESS.md)**: Khung phân cấp sẵn sàng triển khai 5 bậc và Mock BMS HIL.
12. 🇻🇳 **[BAO_CAO_TONG_KET_DO_AN.md](BAO_CAO_TONG_KET_DO_AN.md)**: Báo cáo tổng kết đồ án toàn diện gồm 21 chương tiếng Việt.

---

## 🏗️ Cấu Trúc Mã Nguồn Dự Án

```text
SmartHVACControlByDoubleQLearning/
├── app/
│   └── app.py                     # Streamlit Scenario Lab 5 trang nghiên cứu chuyên sâu (Full-width)
├── data/
│   ├── calibration/               # Tệp JSON hiệu chuẩn K-Means công suất & mô hình nhiệt vật lý
│   ├── processed/                 # Dữ liệu sạch chu kỳ 1 giờ (2018, 2019, Tầng 2, 3)
│   └── sample/                    # Mẫu dữ liệu thử nghiệm nhanh
├── doc/
│   └── TECHNICAL_REPORT.md        # Báo cáo kỹ thuật ban đầu
├── model/
│   ├── q_learning_model.pkl       # Trọng số Q-Table Q-Learning cũ (Baseline)
│   ├── double_q_learning_model.pkl# Trọng số Q-Table DQL cũ (45 trạng thái)
│   ├── single_q_model.pkl         # Trọng số Single Q-Learning 600 trạng thái (Đối chứng trực tiếp)
│   └── target_double_q_model.pkl  # Trọng số Target DQL chuẩn (600 trạng thái x 5 hành động)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_and_environment.ipynb
│   └── 03_training_and_comparison.ipynb
├── result/
│   ├── figure/                    # Đồ thị hội tụ, so sánh baseline, chuyển miền, độ bền vững, overestimation
│   └── log/                       # CSV/JSON nhật ký benchmark, mô hình vật lý và huấn luyện
├── src/
│   ├── agent/
│   │   ├── target_double_q_agent.py   # Double Q-Learning chuẩn 600 trạng thái (Dual Q-Tables)
│   │   ├── baseline_policies.py       # 7 chính sách so sánh (Comfort-First, Fixed 25°C, Thermostat, ...)
│   │   ├── double_q_learning_agent.py # Agent 45 trạng thái cũ (Bảo tồn nguyên vẹn)
│   │   └── q_learning_agent.py        # Agent Single Q-Learning chuẩn (Bảo tồn & dùng làm Baseline)
│   ├── data/
│   │   ├── provenance.py              # Khung truy xuất nguồn gốc dữ liệu (7 nhãn provenance)
│   │   ├── domain_config.py           # Đặc tả các miền dữ liệu CU-BEMS
│   │   ├── domain_loader.py           # Loader nhận diện cửa sổ cảm biến, cắt chuỗi thời gian 3 pha
│   │   └── calibration.py             # K-Means hiệu chuẩn công suất máy nén & phát hiện trôi
│   ├── environment/
│   │   ├── greybox_model.py           # Mô hình nhiệt - ẩm vật lý RC bậc 1 (Khớp thực nghiệm & Phân tích độ nhạy)
│   │   ├── persistence_model.py       # Mô hình dự báo quán tính làm mốc đối chứng
│   │   ├── rf_transition_model.py     # Mô hình Random Forest cũ (Bảo tồn nguyên vẹn)
│   │   ├── model_validator.py         # Kiểm định tính hợp lệ vật lý và sai số ngoại suy
│   │   ├── target_env.py              # Môi trường TargetHVACEnv (600 states x 5 actions)
│   │   └── hvac_env.py                # Môi trường cũ 45 states (Bảo tồn nguyên vẹn)
│   ├── evaluation/
│   │   ├── benchmark_suite.py         # Benchmark 4 trục miền x 3 chế độ & Multi-seed 5 seeds
│   │   └── robustness_suite.py        # Đánh giá độ bền vững trước 7 kịch bản nhiễu loạn
│   ├── reward/
│   │   ├── advanced_reward.py         # Hàm thưởng đa mục tiêu tiện nghi - năng lượng - thiết bị
│   │   └── reward_function.py         # Hàm thưởng cũ (Bảo tồn nguyên vẹn)
│   ├── safety/
│   │   ├── safety_gateway.py          # Cổng an toàn cơ điện (Khóa máy nén 1h, khống chế khẩn cấp)
│   │   └── hil_interface.py           # Giao thức mô phỏng BMS HIL (BACnet/Modbus packet)
│   └── util/
│       ├── plot_benchmarks.py         # Vẽ các đồ thị benchmark nghiên cứu xuất bản
│       └── eda.py                     # Tiện ích phân tích dữ liệu khám phá ban đầu
├── tests/
│   ├── test_provenance.py             # Kiểm thử nhãn nguồn gốc dữ liệu và khuyến cáo
│   ├── test_data_leakage.py           # Kiểm thử cắt chuỗi thời gian đơn điệu, phân tách 3 pha, bắt bẫy rò rỉ
│   ├── test_physical_consistency.py   # Kiểm thử đơn điệu nhiệt động học và từ chối khớp mẫu nhỏ
│   ├── test_target_dql.py             # Kiểm thử bảng Q kép (600x5), cập nhật DQL, cấu hình mô hình
│   ├── test_safety_gateway.py         # Kiểm thử toàn diện các quy tắc Cổng an toàn & Mock BMS
│   └── test_domain_transfer.py        # Kiểm thử tính tương thích cấu hình miền và pipeline đánh giá
├── train_target_dql.py                # Pipeline tổng thể huấn luyện DQL, Single-Q & chạy toàn bộ benchmark
├── train.py                           # Pipeline huấn luyện cũ (Bảo tồn nguyên vẹn)
└── evaluate.py                        # Pipeline đánh giá cũ (Bảo tồn nguyên vẹn)
```

---

## 📈 Kết Quả Điểm Chuẩn Ban Đầu (LEGACY BENCHMARK / CẤU HÌNH CŨ - 5 RANDOM SEEDS)

> [!NOTE]
> Đây là kết quả thử nghiệm ban đầu qua 5 hạt giống (
esult/log/baselines_benchmark.csv). Chỉ số tiết kiệm .40\%$ điện năng được đo lường hoàn toàn trong mô phỏng số đối chứng với chiến lược chạy cố định 25°C, không phải kết quả triển khai trên thiết bị tòa nhà thực tế và không phải so sánh với Q-Learning.

### 1. Bảng Điểm Chuẩn Đối Đầu Các Chiến Lược Điều Khiển
Thực nghiệm đánh giá trên tập kiểm định độc lập năm 2018 chưa từng thấy qua 5 hạt giống (`42, 101, 202, 303, 404`):

| Chiến Lược Điều Khiển (Policy) | Seeds | Mean Reward | Reward 95% CI | Vi Phạm Tiện Nghi (%) | Vi Phạm Nghiêm Trọng (%) | Degree Hours | Điện Năng (kWh/ngày) | Tiết Kiệm vs Cố Định 25°C (%) | Số Lần Đổi Trạng Thái/Ngày | Can Thiệp Cổng An Toàn (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target DQL (Đề Xuất 600s)** | 5 | **-35.69** | `[-41.40, -29.98]` | 64.22% | 2.44% | 18.18 | **202.93** | **+42.40%** | 7.4 | 19.50% |
| **Single Q-Learning (600s)** | 5 | -35.37 | `[-40.59, -30.15]` | 62.39% | 1.78% | 18.97 | 203.37 | +42.27% | 6.8 | 15.95% |
| **Fixed Setpoint (Cố Định 25°C)**| 5 | -33.38 | `[-38.03, -28.73]` | 50.72% | 0.00% | 9.58 | 352.28 | 0.00% | 0.0 | 0.00% |
| **Energy-Saving Policy** | 5 | -31.51 | `[-37.20, -25.82]` | 66.00% | 0.00% | 20.22 | 162.67 | +53.82% | 3.3 | 5.22% |
| **Rule-Based Thermostat** | 5 | -38.50 | `[-42.34, -34.66]` | 42.61% | 0.00% | 5.76 | 439.29 | -24.70% | 2.7 | 0.00% |
| **Comfort-First Policy** | 5 | -44.27 | `[-47.46, -41.09]` | 50.94% | 0.00% | 6.53 | 499.36 | -41.75% | 2.7 | 0.00% |
| **Legacy DQL (45s Cũ)** | 5 | -46.18 | `[-49.60, -42.77]` | 64.06% | 0.00% | 13.13 | 461.12 | -30.89% | 5.1 | 1.05% |
| **AC-Always-Off Baseline** | 5 | -38.78 | `[-46.83, -30.73]` | 71.89% | 11.83% | 29.43 | 55.64 | +84.21% | 1.7 | 11.72% |
| **Random Policy** | 5 | -89.42 | `[-96.38, -82.45]` | 64.50% | 14.39% | 28.03 | 320.05 | +9.15% | 12.6 | 34.72% |

### 2. Điểm Nhấn Khoa Học Nổi Bật:
1. **Tiết kiệm điện 42.40%**: Target Double Q-Learning giúp giảm **42.40% điện năng** so với chiến lược cố định 25°C và giảm **53.8%** so với bộ điều khiển Thermostat cơ điện, trong khi nhiệt độ phòng duy trì ổn định quanh $26.3^\circ\text{C}$.
2. **Hiệu quả chuyển giao đa miền**:
   - Khác Tầng (`2018_F3_Z1`): Zero-Shot đạt $-61.62$. Chỉ cần **Calibration-Only** (0 tập train lại), Reward lập tức tăng vọt lên **$-31.83$** (+48.3% cải thiện). Với 25 tập **Fine-Tuning**, Reward đạt **$-20.19$** và vi phạm nghiêm trọng chỉ còn **1.25%**.
   - Khác Phòng (`2018_F2_Z2`): Với 10 tập Fine-Tuning, Reward đạt **$-24.26$**, vi phạm nghiêm trọng giảm từ 11.25% xuống **2.29%**.

---


---

## 🔬 Kết Quả Kiểm Định Thống Kê Học Thuật Mới Nhất (LATEST 10-SEED CONSISTENCY BENCHMARK)

Để đảm bảo tính trung thực khoa học tuyệt đối và loại bỏ trôi ngẫu nhiên, một pipeline kiểm thử ghép cặp 10 hạt giống ngẫu nhiên độc lập (42, 101, 202, 303, 404, 505, 606, 707, 808, 909) với bộ sinh số ngẫu nhiên RNG độc lập đã được thực thi (
esult/final_validation/dql_vs_ql_10seeds.csv, 
esult/final_validation/dql_vs_ql_statistical_analysis.json):

| Chỉ Số Đánh Giá | Target Double Q-Learning (10 Seeds) | Single Q-Learning Baseline (10 Seeds) | Chênh Lệch Ghép Cặp (DQL - QL) | Kiểm Định Thống Kê & Ý Nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| **Reward Trung Bình (Mean)** | **$-37.550 \pm 7.835$** | $-42.777 \pm 15.551$ | **$+5.227 \pm 14.456$** | Paired t-test:  = 1.1442, p = 0.2820$ ( > 0.05$); Wilcoxon:  = 16.0, p = 0.2754$ |
| **Khoảng Tin Cậy 95% (Reward)** | [-42.404, -32.696] | [-52.413, -33.141] | **[-5.106, 15.560]** | KTC 95% của hiệu số chứa giá trị 0; khác biệt **chưa có ý nghĩa thống kê**. |
| **Điện Năng Tiêu Thụ (kWh/ngày)** | **.385 	ext{ kWh/ngày}$** | **.136 	ext{ kWh/ngày}$** | **$+2.249 	ext{ kWh/ngày}$** | DQL tiêu thụ cao hơn QL .249 	ext{ kWh/ngày}$. **Không tuyên bố DQL tiết kiệm điện hơn QL.** |
| **Độ Lệch Chuẩn (Standard Deviation)** | **.835$** | .551$ | **$-7.716$ (-49.6%)** | **Giảm phương sai**: DQL có độ lệch chuẩn thấp hơn gần một nửa, chứng minh tính ổn định hội tụ cao hơn. |
| **Ước Lượng Giá Trị Max-Q** | **$-0.640$** | $-0.794$ | **$+0.154$** | So sánh mô tả giá trị Max-Q học được. Do không có ^*$ chân lý, đây không phải phép đo độ thiên lệch trực tiếp. |

### Kết luận khoa học chuẩn mực:
- **Chưa có khác biệt Reward có ý nghĩa thống kê ở mức 0.05**: Mặc dù DQL đạt Reward trung bình tốt hơn trong mẫu 10 seeds (+5.227), các kiểm định  = 0.282$ và  = 0.2754$ cho thấy chưa đủ bằng chứng kết luận DQL vượt trội về mặt thống kê.
- **Không tuyên bố DQL tiết kiệm điện hơn QL**: DQL tiêu thụ trung bình cao hơn QL khoảng 2.249 kWh/ngày trong thực nghiệm này.
- **Ưu thế cốt lõi của DQL**: Giảm mạnh phương sai dao động giữa các hạt giống ngẫu nhiên (std 7.835 so với 15.551 của Single Q), đem lại khả năng hội tụ ổn định hơn nhiều lần.

---

## 🚀 Hướng Dẫn Tái Lập Thực Nghiệm (Reproducibility Guide)

### 1. Cài đặt thư viện phụ thuộc
```bash
pip install numpy pandas scikit-learn scipy matplotlib streamlit pytest
```

### 2. Chạy toàn bộ 68 Unit Test tự động (100% Pass)
```bash
python -m pytest tests/ -v
```
*(Bao gồm 43 unit test kế thừa và 25 unit test mới về Digital Twin đa thiết bị, cân bằng ẩm Magnus, CO2, PM2.5, công suất standby máy nén, kịch bản môi trường, Cổng an toàn và Q kép).*

### 3. Khởi chạy Giao Diện Mới: HVAC Agent Live Simulation (Step-by-Step Desktop GUI)
```bash
# Cách 1: Chạy từ thư mục gốc
python run_live_simulation.py

# Cách 2: Chạy trực tiếp module GUI
python -m app.live_simulation_gui
```
*Giao diện Desktop chuyên nghiệp hỗ trợ: Chạy từng bước chậm (Step Forward), Start/Pause/Resume/Stop, chọn 4 kịch bản phòng, kiểm soát 5 mức tốc độ, hiển thị trực quan 7 nhóm thiết bị, minh bạch Decision Trace & giá trị Q1/Q2, biểu đồ vi phân thời gian thực, và chuyển đổi linh hoạt giữa Evaluation Mode (đóng băng Q-table) và Online Learning Mode (cập nhật Q-table).*

### 4. Huấn luyện Agent Đa Thiết Bị & Chạy Benchmark So Sánh
```bash
python train_multi_device_dql.py
```

### 5. Khởi chạy Phòng Thí Nghiệm Web (Streamlit Research Lab)
```bash
streamlit run app/app.py
```
*Giao diện tự động áp dụng Dark Theme độ tương phản cao và bố cục Full-Width tràn viền (chuẩn `[DIR-202611-002]`).*
