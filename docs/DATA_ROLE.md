# VAI TRÒ CỦA CÁC TẬP DỮ LIỆU TRONG ĐÁNH GIÁ (DATA ROLE SPECIFICATION)

Tài liệu này xác định rõ vai trò, mục tiêu và phạm vi sử dụng của từng phân vùng dữ liệu trong đồ án "Smart HVAC Control by Double Q-Learning", bảo đảm tuân thủ nguyên tắc liêm chính khoa học và chống rò rỉ dữ liệu (data leakage).

---

## 1. Bảng Tổng hợp Phân vùng Dữ liệu

| Tên Phân Vùng | Nguồn Dữ Liệu | Tỷ Lệ / Quy Mô | Vùng Đệm (Buffer) | Vai Trò & Mục Đích Chính | Quyền Điều Chỉnh |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `2018_F2_Z1` | 60% đầu (~26.4 ngày) | Không | Huấn luyện chính sách RL, cập nhật Q-table, hiệu chuẩn tham số vật lý | Cho phép cập nhật tham số |
| **Validation** | `2018_F2_Z1` | 20% giữa (~8.8 ngày) | Cách Train 24h | Tìm kiếm siêu tham số ($\alpha, \epsilon_{\text{decay}}, \text{episodes}$), chọn checkpoint tốt nhất | Dùng để chọn siêu tham số |
| **Test-A** | `2018_F2_Z1` | 20% cuối (~8.8 ngày) | Cách Val 24h | Đánh giá tổng quát hóa cùng miền (In-Distribution Generalization) | **ĐÃ KHÓA** — Chạy duy nhất 1 lần |
| **Test-B** | `2019_F2_Z1` | Dữ liệu năm 2019 | Khác năm quan trắc | Đánh giá chuyển dịch miền theo năm (Cross-Year Domain Transfer) | **ĐÃ KHÓA** — Chạy duy nhất 1 lần |
| **Test-C** | Dữ liệu mô phỏng | 3 kịch bản cực đoan | Môi trường tổng hợp | Đánh giá độ bền bỉ trong điều kiện stress thử thách (Stress Testing) | **ĐÃ KHÓA** — Chạy duy nhất 1 lần |

---

## 2. Chi Tiết Vai Trò Của Từng Tập

### 2.1. Tập Huấn Luyện (Train Split)
- **Dữ liệu:** 60% đầu của miền chính `2018_F2_Z1`.
- **Mục tiêu:**
  - Cung cấp môi trường để các thuật toán học tăng cường (Double Q-Learning, Single Q-Learning) khám phá không gian trạng thái - hành động (720 trạng thái $\times$ 12 hành động) và cập nhật bảng giá trị $Q$.
  - Dùng để hiệu chuẩn đúng 3 tham số vật lý của mô hình cuộn lạnh ở Phase 1c: nhiệt độ ra khỏi cuộn $T_{coil} = 12.0^\circ\text{C}$, lưu lượng qua cuộn $V_{coil} = 0.08\text{ m}^3\text{/s/kW}$ và độ ẩm ngoài trời cơ sở $RH_{out} = 68.0\%$.
  - Tính toán các giá trị thống kê chuẩn hóa đặc trưng (Mean, Std) cho toàn bộ pipeline.

### 2.2. Tập Kiểm Tra Xác Thực (Validation Split)
- **Dữ liệu:** 20% tiếp theo của miền chính `2018_F2_Z1`, ngăn cách với tập Train bởi 24 giờ đệm.
- **Mục tiêu:**
  - Khảo sát lưới siêu tham số (Hyperparameter Grid Search) một cách công bằng giữa Double Q-Learning và Single Q-Learning với cùng ngân sách tìm kiếm.
  - Đánh giá hàm mục tiêu kết hợp (phần thưởng tích lũy, số giờ vi phạm tiện nghi nhiệt/độ ẩm/CO2, năng lượng tiêu thụ).
  - Lựa chọn bộ tham số tối ưu duy nhất trước khi chuyển sang giai đoạn kiểm thử cuối cùng.

### 2.3. Tập Kiểm Thử A — Tổng Quát Hóa Cùng Miền (Test-A)
- **Dữ liệu:** 20% thời gian cuối của miền `2018_F2_Z1`, cách tập Validation 24 giờ đệm.
- **Mục tiêu:**
  - Đánh giá khả năng điều khiển của các mô hình đã huấn luyện trên các ngày thời tiết tiếp theo của cùng một năm mà hệ thống chưa từng nhìn thấy trong quá trình học.
  - So sánh trực tiếp 6 bộ điều khiển: Double Q-Learning, Single Q-Learning, Fixed Setpoint, Thermostat, Rule-based, Random Controller.

### 2.4. Tập Kiểm Thử B — Chuyển Dịch Miền Xuyên Năm (Test-B)
- **Dữ liệu:** Tập dữ liệu `2019_F2_Z1` ghi nhận trên cùng một khu vực (Tầng 2, Vùng 1) nhưng trong năm 2019.
- **Mục tiêu:**
  - Đo lường mức độ suy giảm hiệu năng (distributional shift penalty) của bộ điều khiển RL khi áp dụng sang chu kỳ năm kế tiếp có điều kiện thời tiết ngoài trời, mật độ sử dụng và tải nhiệt thay đổi.
  - Kiểm tra xem thuật toán có bị quá khớp (overfitting) với chu kỳ thời tiết của riêng năm 2018 hay không.

### 2.5. Tập Kiểm Thử C — Đánh Giá Sức Chịu Đựng Dưới Điều Kiện Cực Đoan (Test-C)
- **Dữ liệu:** 3 kịch bản tổng hợp mô phỏng điều kiện ứng suất cao (Stress Testing):
  1. `heatwave` (Sóng nhiệt gay gắt): Nhiệt độ ngoài trời tăng đột biến ($T_{out} > 38^\circ\text{C}$ kéo dài), thử thách năng lực giữ tiện nghi nhiệt của hệ thống AC.
  2. `large_occupancy` (Phòng quá tải người): Mật độ người tập trung đông, làm tăng tải nhiệt nội sinh và tốc độ phát thải CO2, thử thách phản xạ thông gió và làm mát đồng thời.
  3. `smog` (Ô nhiễm khói bụi nghiêm trọng): Nồng độ bụi ngoài trời vượt ngưỡng báo động (PM2.5 ngoài trời $> 150\,\mu\text{g/m}^3$), thử thách khả năng điều hòa giữa lọc không khí và thông gió ngoài để không đưa bụi bẩn vào phòng.

---

## 3. Ràng Buộc Liêm Chính Tuyệt Đối

1. **Nguyên tắc "Chỉ nhìn một lần" (Single-Look Rule):** Các tập Test-A, Test-B và Test-C chỉ được đưa vào đánh giá sau khi toàn bộ thuật toán, siêu tham số, trọng số phần thưởng và checkpoint mô hình đã được cố định hoàn toàn (tag git `protocol-locked`).
2. **Không điều chỉnh ngược:** Tuyệt đối không được phép thay đổi siêu tham số, sửa hàm phần thưởng hoặc can thiệp mã nguồn sau khi xem kết quả kiểm thử. Kết quả dù DQL thắng, hòa hay thua QL hoặc baselines đều được giữ nguyên văn và báo cáo trung thực.
