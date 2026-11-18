# HVAC SCOPE & COVERAGE — DIGITAL TWIN BOUNDARIES

> [!IMPORTANT]
> **Tuyên bố Ranh giới Nghiên cứu (`RESEARCH_SIMULATION_ONLY`):**  
> Dự án này tập trung vào việc nghiên cứu thuật toán **Double Q-Learning trong bài toán điều khiển phối hợp đa thiết bị HVAC** tại cấp độ phòng/khu vực (Zone/Room Level). Tài liệu này minh định rõ ràng những gì được mô phỏng trực tiếp, những gì được đơn giản hóa và những gì nằm ngoài phạm vi nghiên cứu để bảo đảm tính trung thực khoa học.

---

## 1. Phân loại 3 Cấp độ Phạm vi Mô phỏng (3-Tier Scope Architecture)

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: DIRECTLY SIMULATED (Mô phỏng Trực tiếp)                        │
│ • Truyền nhiệt nhạy 2R-1C (Nhiệt độ phòng °C)                          │
│ • Cân bằng khối lượng ẩm tâm trắc học Magnus (Độ ẩm %RH)               │
│ • Phát thải CO2 người & Pha loãng thông gió (ppm)                      │
│ • Lọc bụi mịn HEPA CADR & Lắng đọng bề mặt (PM2.5 µg/m³)               │
│ • Công suất tức thời & Điện năng tích lũy 7 nhóm Actuator (kW, kWh)    │
│ • Biểu giá điện theo khung giờ cao điểm (Time-of-Use Tariff)           │
│ • 3 cấu hình kích thước phòng (Small 20m², Medium 50m², Large 120m²)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ TIER 2: SIMPLIFIED / LUMPED (Mô hình Khối Gộp Đơn giản hóa)            │
│ • Không khí xáo trộn đồng nhất (Well-mixed assumption, không dùng CFD)  │
│ • Vỏ bao che đại diện bởi U-envelope và C_room gộp duy nhất            │
│ • Hệ số hiệu quả làm lạnh COP máy nén coi là hằng số trung bình        │
│ • Lưu lượng đối lưu cửa sổ tính qua chênh lệch nhiệt độ tương đối      │
│ • Nguồn thời tiết ngoại cảnh Bangkok theo mô hình tuần hoàn ngày đêm   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ TIER 3: OUT OF SCOPE / EXCLUDED (Nằm Ngoài Phạm vi Đồ án)              │
│ ❌ Trạm máy Chiller trung tâm công nghiệp giải nhiệt nước              │
│ ❌ Tháp giải nhiệt (Cooling Towers) và Nồi hơi (Boilers)               │
│ ❌ Động lực học chất lưu 3D, phân tầng nhiệt và dòng xoáy cục bộ (CFD) │
│ ❌ Tổn thất áp suất khí động học đường ống gió và van chặn lửa         │
│ ❌ Giao thức truyền thông vật lý phần cứng (BACnet IP, Modbus RTU)     │
│ ❌ Triển khai thực tế trên hệ thống HVAC vật lý của tòa nhà            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết Từng Cấp độ

### 2.1. Cấp độ 1 — Mô phỏng Trực tiếp (Directly Simulated)
- **7 Nhóm Thiết bị Độc lập:** Máy lạnh (AC), Máy sưởi (Heater), Quạt cấp/hút thông gió (Ventilation), Cửa sổ/van gió tự nhiên (Window), Máy tạo ẩm (Humidifier), Máy hút ẩm (Dehumidifier), Máy lọc không khí HEPA (Filter).
- **Phản hồi Đơn điệu (Monotonicity):** Tăng công suất làm lạnh làm giảm nhiệt độ; tăng quạt thông gió làm giảm nồng độ $CO_2$; bật lọc HEPA làm giảm nồng độ $PM_{2.5}$.
- **Tương tác Vật lý Đa biến:** Làm lạnh máy nén ngưng tụ hơi ẩm phụ trợ trên dàn lạnh; thông gió tươi mang theo nhiệt và ẩm ngoài trời vào phòng; người sinh đồng thời nhiệt nhạy, hơi ẩm và khí $CO_2$.

### 2.2. Cấp độ 2 — Đơn giản hóa Có căn cứ (Simplified / Lumped Approximations)
- **Mô hình Khối tập trung (Lumped Capacitance):** Không gian phòng được xem như một thể tích đồng nhất về nhiệt độ và nồng độ chất khí, không mô phỏng hiện tượng phân tầng nhiệt từ sàn lên trần.
- **Tải Bức xạ Mặt trời Gộp:** Hiệu ứng bức xạ mặt trời qua kính (SHGC) được gộp chung vào điện trở nhiệt vỏ bao che tương đương $R_{env} = \frac{1}{U_{env} \cdot A_{env}}$.
- **Hiệu suất COP Cố định:** Máy lạnh sử dụng COP danh định trung bình 3.2, không mô hình hóa sự suy giảm COP theo đường cong nhiệt độ ngưng tụ ngoài trời.

### 2.3. Cấp độ 3 — Nằm Ngoài Phạm vi (Strictly Out of Scope)
- Hệ thống trung tâm phức tạp (Central AHU, Water Chillers, Cooling Towers, Chilled Water Pumps) thuộc về các công trình công nghiệp quy mô lớn và không phù hợp với mục tiêu nghiên cứu thuật toán học tăng cường dạng bảng tại khu vực phòng đơn/phòng làm việc.
- Việc kết nối và điều khiển trực tiếp trên thiết bị HVAC thật đòi hỏi quy trình kiểm thử phần cứng HIL (Hardware-in-the-Loop) công nghiệp, xin cấp phép an toàn tòa nhà và hạ tầng cảm biến chuẩn công nghiệp, điều nằm ngoài khuôn khổ đồ án nghiên cứu lý thuyết và mô phỏng này.

---

## 3. Tóm tắt Đánh giá Ranh giới

Việc giới hạn phạm vi mô phỏng ở Cấp độ 1 và Cấp độ 2 cho phép:
1. Giữ môi trường tính toán đủ nhanh để huấn luyện hàng trăm episode học tăng cường trong vài giây.
2. Bảo toàn tính giải thích được (interpretability) của các phương trình vật lý cơ sở.
3. Không tạo ra các tuyên bố phóng đại sai lệch về năng lực triển khai thực tế.
