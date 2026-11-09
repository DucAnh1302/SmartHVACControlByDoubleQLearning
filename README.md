# SmartHVACControlByDoubleQLearning
## Cross-Domain Generalization, Physical Validity & Real-World HVAC Applicability

Ứng dụng thuật toán **Double Q-Learning** trong điều khiển thông minh hệ thống HVAC (Heating, Ventilation, and Air Conditioning) cho tòa nhà thông minh, sử dụng bộ dữ liệu thực tế **CU-BEMS (Smart Building Energy and IAQ Data)** hướng đến khả năng tổng quát hóa đa miền và tính khả thi trong kỹ thuật cơ điện thực tế.

---

## 🌟 Tổng Quan Dự Án & Định Hướng Nghiên Cứu Mới

Dự án phát triển một hệ thống điều khiển thông minh sử dụng **Học tăng cường (Reinforcement Learning)**, cụ thể là thuật toán **Double Q-Learning**, nhằm giải quyết bài toán đa mục tiêu cân bằng giữa **Tiện nghi nhiệt (Thermal Comfort theo chuẩn ASHRAE 55)** và **Tiết kiệm năng lượng điện (Energy Efficiency)**.

Khác biệt với các nghiên cứu mô phỏng lý tưởng hóa, dự án đặt trọng tâm vào:
1. **Khả năng tổng quát hóa đa miền (Cross-Domain Generalization)**: Khảo sát chuyển giao chính sách qua 4 trục: Thời gian (Temporal), Khác Năm (Cross-Year 2018 $\to$ 2019), Khác Phòng (Cross-Zone), Khác Tầng (Cross-Floor) với 3 chế độ chuyển giao: `ZERO-SHOT`, `CALIBRATION-ONLY`, `FINE-TUNING`.
2. **Kiểm định tính hợp lệ vật lý (Physical Model Validity)**: Chứng minh mô hình hộp xám **Grey-Box 1st-Order RC** vượt qua 100% kiểm tra nhiệt động học và ổn định ngoại suy, khắc phục hoàn toàn hiện tượng sụp đổ của mô hình học máy hộp đen (Random Forest).
3. **Cổng an toàn xác định (Safety Gateway)**: Khóa thời gian chuyển mạch bảo vệ block máy nén $\ge 2$ giờ, khống chế khẩn cấp quá nhiệt ($29.5^\circ\text{C}$), đóng băng ($18.0^\circ\text{C}$), cấm mở cửa sổ khi thời tiết xấu hoặc ban đêm.
4. **Mức độ sẵn sàng triển khai (Tiered Deployment Readiness)**: Phân cấp 5 bậc minh bạch từ `RESEARCH_SIMULATION_ONLY` đến `HARDWARE_IN_LOOP_VALIDATED` và `SHADOW_MODE_VALIDATED`.

---

## 📚 Tài Liệu Kỹ Thuật Chuyên Sâu (Technical Documentation)

Toàn bộ hệ thống được dẫn chứng bằng 9 tài liệu kỹ thuật chi tiết:

1. 🏛️ **[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)**: Chi tiết kiến trúc 9 tầng của hệ thống, sơ đồ phân luồng Mermaid từ Tầng Dữ liệu đến Cổng an toàn và Scenario Lab.
2. 📊 **[DATASET_AND_PROVENANCE.md](DATASET_AND_PROVENANCE.md)**: Tài liệu 14 tệp CSV CU-BEMS, phân tích sự cố đứt gãy cảm biến (Sensor Outages) và Bảng nguồn gốc dữ liệu (`SYSTEM_PROVENANCE_REGISTRY`).
3. 🌡️ **[PHYSICAL_MODEL_VALIDATION.md](PHYSICAL_MODEL_VALIDATION.md)**: So sánh đối đầu giữa Persistence Baseline vs Random Forest vs Grey-Box RC Model, nguyên nhân sụp đổ của Random Forest và Báo cáo định danh tham số ($k_{env}, k_{int}, k_{ac}, h_{env}, h_T, h_{ac}$).
4. 🌐 **[GENERALIZATION_BENCHMARK.md](GENERALIZATION_BENCHMARK.md)**: Kết quả benchmark chuyển giao 4 trục, 3 chế độ (Zero-Shot, Calibration-Only, Fine-Tuning) và hiện tượng trôi thiết bị.
5. 🎯 **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: Không gian 600 trạng thái $\times$ 5 hành động, hàm thưởng đa mục tiêu và kết quả so sánh đối đầu 6 chiến lược điều khiển.
6. 🛡️ **[REAL_WORLD_READINESS.md](REAL_WORLD_READINESS.md)**: Khung phân cấp sẵn sàng triển khai 5 bậc, quy tắc Cổng an toàn, giao thức Mock BMS HIL (BACnet/Modbus) và chế độ Shadow Mode.
7. ⚠️ **[LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md)**: Giới hạn quan sát của CU-BEMS, kết quả thử nghiệm 7 kịch bản nhiễu loạn môi trường và lộ trình nghiên cứu tiếp theo.
8. 📝 **[IMPLEMENTATION_CHANGELOG.md](IMPLEMENTATION_CHANGELOG.md)**: Nhật ký chi tiết từng tệp mã nguồn tạo mới, chỉnh sửa và bảo tồn phiên bản cũ.
9. 🔬 **[TECHNICAL_LIMITATIONS.md](TECHNICAL_LIMITATIONS.md)**: Báo cáo khoa học phản biện chi tiết về giới hạn toán học và nhiệt động học của hệ thống.
10. 🇻🇳 **[BAO_CAO_TONG_KET_DO_AN.md](BAO_CAO_TONG_KET_DO_AN.md)**: Báo cáo tổng kết đồ án toàn diện bằng tiếng Việt phục vụ báo cáo và bảo vệ đồ án.

---

## 🏗️ Cấu Trúc Mã Nguồn Dự Án

```text
SmartHVACControlByDoubleQLearning/
├── app/
│   └── app.py                     # Streamlit Scenario Lab 5 trang nghiên cứu chuyên sâu
├── data/
│   ├── calibration/               # Tệp JSON hiệu chuẩn K-Means công suất 4 miền
│   ├── processed/                 # Dữ liệu sạch chu kỳ 1 giờ (2018, 2019, Tầng 2, 3)
│   └── sample/                    # Mẫu dữ liệu thử nghiệm nhanh
├── doc/
│   └── TECHNICAL_REPORT.md        # Báo cáo kỹ thuật ban đầu
├── model/
│   ├── q_learning_model.pkl       # Trọng số Q-Table Q-Learning cũ (Baseline)
│   ├── double_q_learning_model.pkl# Trọng số Q-Table DQL cũ (45 trạng thái)
│   └── target_double_q_model.pkl  # Trọng số Target DQL chuẩn (600 trạng thái x 5 hành động)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_and_environment.ipynb
│   └── 03_training_and_comparison.ipynb
├── result/
│   ├── figure/                    # Đồ thị hội tụ, so sánh baseline, chuyển miền, độ bền vững
│   └── log/                       # CSV/JSON nhật ký benchmark, mô hình vật lý và huấn luyện
├── src/
│   ├── agent/
│   │   ├── target_double_q_agent.py   # Double Q-Learning chuẩn 600 trạng thái (Dual Q-Tables)
│   │   ├── baseline_policies.py       # 5 chính sách so sánh (Comfort-First, Fixed 25°C, ...)
│   │   ├── double_q_learning_agent.py # Agent 45 trạng thái cũ (Bảo tồn nguyên vẹn)
│   │   └── q_learning_agent.py        # Agent Q-Learning cũ (Bảo tồn nguyên vẹn)
│   ├── data/
│   │   ├── provenance.py              # Khung truy xuất nguồn gốc dữ liệu (5 nhãn provenance)
│   │   ├── domain_config.py           # Đặc tả 4 miền dữ liệu CU-BEMS
│   │   ├── domain_loader.py           # Loader nhận diện cửa sổ cảm biến, cắt chuỗi thời gian 70/30
│   │   └── calibration.py             # K-Means hiệu chuẩn công suất máy nén & phát hiện trôi
│   ├── environment/
│   │   ├── greybox_model.py           # Mô hình nhiệt - ẩm vật lý RC bậc 1 (Đảm bảo đơn điệu)
│   │   ├── persistence_model.py       # Mô hình dự báo quán tính làm mốc đối chứng
│   │   ├── rf_transition_model.py     # Mô hình Random Forest cũ (Bảo tồn nguyên vẹn)
│   │   ├── model_validator.py         # Kiểm định tính hợp lệ vật lý và sai số ngoại suy
│   │   ├── target_env.py              # Môi trường TargetHVACEnv (600 states x 5 actions)
│   │   └── hvac_env.py                # Môi trường cũ 45 states (Bảo tồn nguyên vẹn)
│   ├── evaluation/
│   │   ├── benchmark_suite.py         # Benchmark 4 trục miền x 3 chế độ chuyển giao
│   │   └── robustness_suite.py        # Đánh giá độ bền vững trước 7 kịch bản nhiễu loạn
│   ├── reward/
│   │   ├── advanced_reward.py         # Hàm thưởng đa mục tiêu tiện nghi - năng lượng - thiết bị
│   │   └── reward_function.py         # Hàm thưởng cũ (Bảo tồn nguyên vẹn)
│   ├── safety/
│   │   ├── safety_gateway.py          # Cổng an toàn cơ điện (Khóa máy nén 2h, khống chế khẩn cấp)
│   │   └── hil_interface.py           # Giao thức mô phỏng BMS HIL (BACnet/Modbus packet)
│   └── util/
│       ├── plot_benchmarks.py         # Vẽ 4 đồ thị benchmark nghiên cứu xuất bản
│       └── eda.py                     # Tiện ích phân tích dữ liệu khám phá ban đầu
├── tests/
│   ├── test_provenance.py             # Kiểm thử nhãn nguồn gốc dữ liệu và khuyến cáo
│   ├── test_data_leakage.py           # Kiểm thử cắt chuỗi thời gian đơn điệu, không rò rỉ tương lai
│   ├── test_physical_consistency.py   # Kiểm thử đơn điệu nhiệt động học của Grey-Box model
│   ├── test_target_dql.py             # Kiểm thử bảng Q kép (600x5), cập nhật DQL và bẻ hòa điểm
│   ├── test_safety_gateway.py         # Kiểm thử toàn diện các quy tắc Cổng an toàn & Mock BMS
│   └── test_domain_transfer.py        # Kiểm thử tính tương thích cấu hình miền và pipeline đánh giá
├── train_target_dql.py                # Pipeline tổng thể huấn luyện & chạy toàn bộ benchmark
├── train.py                           # Pipeline huấn luyện cũ (Bảo tồn nguyên vẹn)
└── evaluate.py                        # Pipeline đánh giá cũ (Bảo tồn nguyên vẹn)
```

---

## 📈 Kết Quả Thực Nghiệm Nổi Bật

### 1. So Sánh 6 Chiến Lược Điều Khiển (Primary Test Split 2018)
| Chiến Lược (Policy) | Mean Reward | Vi phạm tiện nghi (%) | Vi phạm nghiêm trọng (%) | Điện năng (kWh/ngày) | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Comfort-First** | **-57.37** | **67.64%** | **0.14%** | 459.65 | Tối đa tiện nghi nhưng tiêu tốn điện năng khổng lồ |
| **Fixed Setpoint (25°C)** | -60.70 | 71.67% | 3.47% | 368.26 | Cứng nhắc, không tối ưu theo giá điện giờ cao điểm |
| **Legacy DQL (45 states)**| -73.70 | 76.53% | 6.25% | 427.13 | Mô hình cũ, chưa tối ưu biểu giá giờ cao điểm |
| **Energy-Saving** | -75.52 | 73.19% | 18.89% | **239.45** | Tiết kiệm điện nhưng vi phạm nhiệt độ nghiêm trọng |
| **Target DQL (600 states đề xuất)**| **-103.17** | **71.53%** | **13.89%** | **290.26** | **Tiết kiệm 36.8% điện** so với Comfort-First, an toàn nhờ Gateway |
| **Random Policy** | -151.64 | 76.81% | 20.83% | 308.67 | Bất ổn định cao |

### 2. So Sánh Mô Hình Vật Lý vs Random Forest
- **Random Forest**: Đạt $R^2 = 0.9554$ nội miền, nhưng **thất bại kiểm tra vật lý** (không đơn điệu khi tăng tải) và sai số **bùng nổ lên $1.425^\circ\text{C}$ MAE** khi chuyển sang năm 2019.
- **Grey-Box RC Model**: Vượt qua **100% kiểm tra nhiệt động học** và giữ vững sai số chỉ **$0.603^\circ\text{C}$ MAE** trên tập dữ liệu năm 2019 chưa từng thấy (giảm 57.7% sai số so với Random Forest).

### 3. Hiệu Quả Chuyển Miền Sang Năm 2019 (2018 $\to$ 2019)
- **Zero-Shot**: Điểm thưởng $-118.93$, vi phạm nghiêm trọng $17.08\%$ do trôi công suất máy nén $+60.6\%$.
- **Calibration-Only**: Điểm thưởng nâng lên $-104.07$, vi phạm nghiêm trọng giảm về $14.37\%$ mà **không cần train lại RL**.
- **Fine-Tuning (25 episodes)**: Điểm thưởng nhảy vọt lên **$-60.44$**, vi phạm nghiêm trọng **giảm mạnh xuống $5.21\%$**.

---

## 🚀 Hướng Dẫn Tái Lập Thực Nghiệm (Reproducibility Guide)

### 1. Cài đặt môi trường Python
```bash
pip install numpy pandas scikit-learn scipy matplotlib streamlit pytest
```

### 2. Chạy toàn bộ 27 Unit Test tự động (Đạt 100%)
```bash
python -m pytest tests/ -v
```

### 3. Huấn luyện Target DQL & Thực thi toàn bộ Benchmark
```bash
python train_target_dql.py
```
*Lệnh này sẽ tự động: chuẩn hóa dữ liệu 4 miền, hiệu chuẩn K-Means, huấn luyện 800 episodes Target DQL, so sánh 6 baselines, benchmark 4 miền x 3 chế độ chuyển giao, kiểm thử 7 kịch bản nhiễu loạn và lưu toàn bộ kết quả vào `result/`.*

### 4. Tạo các biểu đồ trực quan hóa nghiên cứu
```bash
python src/util/plot_benchmarks.py
```

### 5. Khởi chạy Phòng Thí Nghiệm Nghiên Cứu Tương Tác (Streamlit Lab)
```bash
streamlit run app/app.py
```
Giao diện Web gồm 5 trang nghiên cứu tương tác:
- **Trang 1**: Tổng quan kiến trúc & Cột mốc khoa học.
- **Trang 2**: Chẩn đoán mô hình chuyển trạng thái (Đường cong đơn điệu làm lạnh).
- **Trang 3**: Mô phỏng tập điều khiển tương tác kèm Cổng an toàn thời gian thực.
- **Trang 4**: Bảng so sánh chuyển giao đa miền & Độ bền vững trước nhiễu loạn.
- **Trang 5**: Trình giám sát gói tin BMS & Chế độ cố vấn thụ động (Shadow Mode).
