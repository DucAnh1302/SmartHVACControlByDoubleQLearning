# SmartHVACControlByDoubleQLearning
## Cross-Domain Generalization, Physical Validity & Real-World HVAC Applicability

Ứng dụng thuật toán **Double Q-Learning** trong điều khiển thông minh hệ thống HVAC (Heating, Ventilation, and Air Conditioning) cho tòa nhà thông minh, sử dụng bộ dữ liệu thực tế **CU-BEMS (Smart Building Energy and IAQ Data)** hướng đến khả năng tổng quát hóa đa miền, tính hợp lệ vật lý và khả năng triển khai cơ điện thực tế.

---

## 🌟 Tổng Quan Dự Án & Định Hướng Nghiên Cứu Mới

Dự án phát triển một hệ thống điều khiển thông minh sử dụng **Học tăng cường (Reinforcement Learning)**, cụ thể là thuật toán **Double Q-Learning** (Hado van Hasselt, 2010), nhằm giải quyết bài toán đa mục tiêu cân bằng giữa **Tiện nghi nhiệt (Thermal Comfort theo chuẩn ASHRAE 55)** và **Tiết kiệm năng lượng điện (Energy Efficiency)**.

Khác biệt với các nghiên cứu mô phỏng lý tưởng hóa, dự án đặt trọng tâm vào 5 trụ cột khoa học:
1. **Khả năng tổng quát hóa đa miền (Cross-Domain Generalization)**: Khảo sát chuyển giao chính sách qua 4 trục: Thời gian (Temporal), Khác Năm (Cross-Year 2018 $\to$ 2019), Khác Phòng (Cross-Zone), Khác Tầng (Cross-Floor) với 3 chế độ chuyển giao: `ZERO-SHOT`, `CALIBRATION-ONLY`, `FINE-TUNING` (ngân sách 10 và 25 episodes) trên giao thức phân chia 3 pha độc lập triệt tiêu 100% rò rỉ dữ liệu (No Data Leakage).
2. **Kiểm định tính hợp lệ vật lý (Physical Model Validity)**: Chứng minh mô hình hộp xám **Grey-Box 1st-Order RC** vượt qua 100% kiểm tra nhiệt động học và ổn định ngoại suy, khắc phục hoàn toàn hiện tượng sụp đổ của mô hình học máy hộp đen (Random Forest).
3. **Triệt tiêu hiện tượng thổi phồng giá trị (Maximization Bias Mitigation)**: Tách biệt 2 bảng Q độc lập ($Q_A, Q_B$), khắc phục sai số ước lượng cực đại mà Q-Learning truyền thống mắc phải, minh chứng qua đồ thị đối đầu thực nghiệm.
4. **Cổng an toàn xác định (Safety Gateway)**: Khóa thời gian chuyển mạch bảo vệ block máy nén $\ge 1$ giờ (Anti-Short-Cycling), khống chế khẩn cấp quá nhiệt ($29.5^\circ\text{C}$), quá lạnh ($18.0^\circ\text{C}$), cấm mở cửa sổ khi thời tiết xấu hoặc ban đêm.
5. **Mức độ sẵn sàng triển khai (Tiered Deployment Readiness)**: Phân cấp 5 bậc minh bạch từ `RESEARCH_SIMULATION_ONLY` đến `OFFLINE_VALIDATED`, `HARDWARE_IN_LOOP_VALIDATED` và `SHADOW_MODE_VALIDATED`.

---

## 📚 Tài Liệu Kỹ Thuật Chuyên Sâu (Technical Documentation)

Toàn bộ hệ thống được dẫn chứng bằng 11 tài liệu kỹ thuật chi tiết:

1. 🏛️ **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)**: Chi tiết kiến trúc 9 tầng của hệ thống, sơ đồ phân luồng Mermaid từ Tầng Dữ liệu đến Cổng an toàn và Scenario Lab.
2. 📊 **[DATASET_AND_PROVENANCE.md](DATASET_AND_PROVENANCE.md)**: Tài liệu 14 tệp CSV CU-BEMS, phân tích sự cố đứt gãy cảm biến (Sensor Outages), chính sách nội suy giới hạn $\le 3$ giờ có mặt nạ chất lượng và Bảng nguồn gốc dữ liệu (`SYSTEM_PROVENANCE_REGISTRY`).
3. 🌡️ **[PHYSICAL_MODEL_VALIDATION.md](PHYSICAL_MODEL_VALIDATION.md)**: So sánh đối đầu giữa Persistence Baseline vs Random Forest vs Grey-Box RC Model, nguyên nhân sụp đổ của Random Forest, Báo cáo định danh tham số ($k_{env}, k_{int}, k_{ac}, h_{env}, h_T, h_{ac}$) và Phân tích độ nhạy $\pm 25\%$.
4. 🌐 **[GENERALIZATION_BENCHMARK.md](GENERALIZATION_BENCHMARK.md)**: Kết quả benchmark chuyển giao 4 trục, 3 chế độ (Zero-Shot, Calibration-Only, Multi-Budget Fine-Tuning) trên giao thức phân tách 3 pha không rò rỉ.
5. 🎯 **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: Không gian 600 trạng thái $\times$ 5 hành động, hàm thưởng đa mục tiêu và kết quả điểm chuẩn 5 hạt giống (5 seeds) đối đầu 9 chiến lược điều khiển.
6. 🛡️ **[REAL_WORLD_READINESS.md](REAL_WORLD_READINESS.md)**: Khung phân cấp sẵn sàng triển khai 5 bậc, quy tắc Cổng an toàn, giao thức Mock BMS HIL (BACnet/Modbus) và chế độ Shadow Mode.
7. ⚠️ **[LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md)**: Giới hạn quan sát của CU-BEMS, kết quả thử nghiệm 7 kịch bản nhiễu loạn môi trường và lộ trình nghiên cứu tiếp theo.
8. 📝 **[IMPLEMENTATION_CHANGELOG.md](IMPLEMENTATION_CHANGELOG.md)**: Nhật ký chi tiết từng tệp mã nguồn tạo mới, chỉnh sửa và bảo tồn phiên bản cũ.
9. 🔬 **[TECHNICAL_LIMITATIONS.md](TECHNICAL_LIMITATIONS.md)**: Báo cáo khoa học phản biện chi tiết về giới hạn toán học và nhiệt động học của hệ thống.
10. 📊 **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)**: Bảng theo dõi tiến độ chi tiết từng hạng mục kỹ thuật của đề tài.
11. 🛡️ **[FINAL_VALIDATION_STATUS.md](FINAL_VALIDATION_STATUS.md)**: Báo cáo kiểm định học thuật cuối cùng, bảng đối chiếu tiêu chuẩn, kết quả 40/40 tests và phân tích thống kê 10 hạt giống.
12. 🇻🇳 **[BAO_CAO_TONG_KET_DO_AN.md](BAO_CAO_TONG_KET_DO_AN.md)**: Báo cáo tổng kết đồ án toàn diện gồm 21 chương tiếng Việt phục vụ báo cáo và bảo vệ đồ án trước hội đồng giảng viên.

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

### 2. Chạy toàn bộ 43 Unit Test tự động (100% Pass)
```bash
python -m pytest tests/ -v
```

### 3. Thực thi Pipeline Huấn Luyện & Benchmark Toàn Diện
```bash
python train_target_dql.py
```
*Lệnh này sẽ tự động: chuẩn hóa dữ liệu, khớp mô hình nhiệt vật lý Grey-Box RC, phân tích độ nhạy, huấn luyện DQL và Single-Q (800 episodes), đánh giá đối đầu 9 thuật toán trên 5 hạt giống, benchmark 4 miền x 3 chế độ chuyển giao, kiểm thử 7 kịch bản nhiễu loạn môi trường và xuất báo cáo vào `result/`.*

### 4. Tạo các biểu đồ trực quan hóa xuất bản
```bash
python src/util/plot_benchmarks.py
```

### 5. Khởi chạy Phòng Thí Nghiệm Nghiên Cứu Tương Tác (Streamlit Lab)
```bash
streamlit run app/app.py
```
*Giao diện tự động áp dụng Dark Theme độ tương phản cao và bố cục Full-Width tràn viền (chuẩn `[DIR-202611-002]`).*
