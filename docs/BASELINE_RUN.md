# BÁO CÁO MỐC HIỆN TRẠNG (BASELINE RUN REPORT)

> **Mục đích**: Ghi nhận toàn bộ thông số, kết quả chạy và kiểm thử trên mã nguồn gốc trước bất kỳ can thiệp kỹ thuật nào (Phase 0). Mọi sửa đổi ở các Phase sau sẽ đối chiếu với mốc này.

- **Thời điểm ghi nhận**: 24/11/2026
- **Môi trường thực thi**:
  - Python: `3.12.5` (Windows AMD64)
  - Thư viện chính: `numpy 2.5.4`, `pandas 3.1.1`, `scipy 1.18.1`, `matplotlib 3.11.2`, `pytest 9.2.0`, `python-docx 1.2.0`, `python-pptx 1.0.2`, `openpyxl 3.1.5`
- **Git Commit gốc**: `ba73e37` (Tag: `baseline-goc`, Branch: `huong-a`)
- **Bản sao lưu kết quả gốc**: `result/_baseline_goc/`
  - `result/_baseline_goc/multi_device_benchmark.json`
  - `result/_baseline_goc/multi_device_training_curves.png`
  - `result/_baseline_goc/multi_device_double_q_model.pkl`
  - `result/_baseline_goc/multi_device_single_q_model.pkl`

---

## 1. Kết quả kiểm thử tự động (Pytest)

- **Lệnh thực thi**: `pytest tests -v`
- **Kết quả tổng quát**: **68 passed**, 300 warnings trong 7.87s (100% pass)
- **Cảnh báo**: 300 cảnh báo `Pandas4Warning: Timestamp.dayofweek is deprecated` tại `tests/test_target_env_reset.py:27`.

### Phân bổ 68 test case thực tế theo file:
| STT | Tên file test | Số test pass | Nội dung kiểm tra |
|---|---|:---:|---|
| 1 | `tests/test_advanced_reward.py` | 3 | Phân rã phần thưởng, phạt an toàn độc lập, hồ sơ trọng số |
| 2 | `tests/test_data_leakage.py` | 5 | Chia chuỗi thời gian, chống rò rỉ tương lai, nội suy có chặn |
| 3 | `tests/test_domain_transfer.py` | 2 | Tính nhất quán đặc tả domain, đánh giá mock env |
| 4 | `tests/test_multi_device_agent.py` | 6 | Khởi tạo 720x12, bảng Q kết hợp, cập nhật Double Q, checkpoint |
| 5 | `tests/test_multi_device_reward.py` | 3 | Phân rã cộng tính, hồ sơ ưu tiên, trọng số giờ làm việc |
| 6 | `tests/test_multi_device_safety.py` | 6 | Can thiệp quá nhiệt, đóng băng, CO2, cửa sổ, chống đóng ngắt AC |
| 7 | `tests/test_multi_device_simulation.py` | 10 | Ẩm trắc, làm lạnh đơn điệu, CO2, PM2.5, bảo toàn năng lượng, state encoder |
| 8 | `tests/test_physical_consistency.py` | 6 | Làm lạnh đơn điệu, nhiệt nội sinh, truyền nhiệt vỏ bọc, chặn nhiệt độ an toàn |
| 9 | `tests/test_provenance.py` | 6 | Phủ đầy đủ nhãn nguồn gốc (REAL, DERIVED, SIMULATED), xuất bảng markdown |
| 10 | `tests/test_q_learning_reproducibility.py` | 3 | Tái lập với cùng seed, đa dạng hóa khác seed, cách ly np.random |
| 11 | `tests/test_safety_gateway.py` | 8 | Bộ lọc an toàn độc lập, khóa chống short-cycle, bất biến cấu hình |
| 12 | `tests/test_target_dql.py` | 7 | Khởi tạo agent target, Q kết hợp, luật cập nhật, decay epsilon |
| 13 | `tests/test_target_env_reset.py` | 3 | Cách ly reset episode, không phạt đổi action ở bước 0, tái lập seed |
| **Tổng cộng** | **13 file** | **68** | **Tất cả đều PASS** |

*(Ghi chú: Khắc phục phát hiện F19 - Báo cáo cũ ghi phân bổ sai 18/14/12/10/8/6 vào các file không tồn tại).*

---

## 2. Kết quả huấn luyện & Benchmark thực tế (`train_multi_device_dql.py`)

### 2.1. Quá trình huấn luyện (400 episodes)
- **Độ phủ State-Action (Coverage)**: Đạt **16.1%** sau 400 episodes (chỉ thăm một phần nhỏ trong $720 \times 12 = 8,640$ cặp state-action).
- **Thang đo giá trị Q (Avg Max-Q)**: Trôi đều xuống **-3.56** (do phần thưởng âm chiếm ưu thế).
- **Phần thưởng trung bình (Avg Reward)**: Dao động mạnh trong khoảng **-978 đến -2831** (không có dấu hiệu hội tụ ổn định phẳng).

### 2.2. Kết quả Benchmark 6 bộ điều khiển (15 episodes $\times$ 24 bước = 360 bước đánh giá)
Dữ liệu trích xuất trực tiếp từ `result/multi_device_benchmark.json`:

| Bộ điều khiển | Phần thưởng TB ($\pm$ std) | Điện năng TB (kWh/ngày) | Vi phạm Nhiệt (%) | Vi phạm CO₂ (%) | Can thiệp Gateway (lần/360) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Double Q-Learning** | -408.89 $\pm$ 413.09 | 41.15 $\pm$ 12.75 | 55.56% | 0.56% | 142 (39.4%) |
| **Single Q-Learning** | -548.12 $\pm$ 561.61 | 41.81 $\pm$ 11.70 | 49.17% | 1.67% | 88 (24.4%) |
| **Fixed Setpoint (25°C)** | **-56.18 $\pm$ 56.11** | **39.20 $\pm$ 14.90** | **47.78%** | **0.00%** | **26 (7.2%)** |
| **Conventional Thermostat** | -235.55 $\pm$ 281.55 | 38.30 $\pm$ 15.58 | 57.78% | 9.72% | 38 (10.6%) |
| **Rule-Based Smart HVAC** | -219.50 $\pm$ 248.59 | 42.01 $\pm$ 14.16 | 53.61% | 0.00% | 15 (4.2%) |
| **Random Policy** | -433.74 $\pm$ 488.42 | 40.94 $\pm$ 14.12 | 78.61% | 1.11% | 170 (47.2%) |

### 2.3. Nhận định cốt lõi từ số liệu thật (Đối chiếu F1, F2, F5, F7):
1. **DQL chưa vượt trội**: DQL (-408.89) chỉ nhỉnh hơn Single Q (-548.12) và Random (-433.74), nhưng thua xa **Fixed Setpoint (-56.18)** và **Rule-Based (-219.50)**.
2. **Tiết kiệm điện là ngộ nhận trong báo cáo cũ**: Fixed Setpoint tiêu thụ 39.2 kWh, thấp hơn DQL (41.2 kWh). Khẳng định "DQL tiết kiệm 42.5% so với Fixed (83.5 kWh)" trong slide/Word cũ là hoàn toàn không có thực.
3. **Lỗi vi phạm nhiệt độ cao ở mọi bộ điều khiển (~48% - 58%)**: Do mất cân đối tải nhiệt nội sinh và công suất điều hòa (F5) cộng với việc env chưa cập nhật occupancy theo từng bước (F3).
4. **Gateway ghi đè nặng**: DQL bị Safety Gateway ghi đè tới 142/360 bước (39.4%) do Agent học mà không có Gateway trong vòng lặp huấn luyện (F7).
