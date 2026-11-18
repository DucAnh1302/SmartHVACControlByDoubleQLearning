# HVAC ACTION SPACE & MULTI-DEVICE COORDINATION ARCHITECTURE

> [!IMPORTANT]
> **Tuyên bố Phạm vi Nghiên cứu (`RESEARCH_SIMULATION_ONLY`):**  
> Kiến trúc phối hợp hành động trong tài liệu này được thiết kế phục vụ thuật toán học tăng cường dạng bảng (Tabular Reinforcement Learning) trong mô phỏng Digital Twin. Việc trừu tượng hóa thành 12 Action Templates là giải pháp nghiên cứu nhằm kiểm soát độ phức tạp mẫu, **không thay thế tiêu chuẩn an toàn cơ điện BACnet/Modbus của hệ thống HVAC vật lý**.

---

## 1. Bài toán Bùng nổ Tổ hợp Hành động (Combinatorial Explosion)

Khi mở rộng bài toán điều khiển từ 1 máy lạnh duy nhất lên 7 nhóm thiết bị HVAC độc lập:
1. Máy làm lạnh (AC): 4 trạng thái (Tắt, Thấp, Trung bình, Cao)
2. Máy sưởi (Heater): 3 trạng thái (Tắt, Thấp, Cao)
3. Quạt thông gió (Ventilation): 4 trạng thái (Tắt, Thấp, Trung bình, Cao)
4. Cửa sổ tự động (Window): 2 trạng thái (Đóng, Mở)
5. Máy tạo ẩm (Humidifier): 4 trạng thái (Tắt, Thấp, Trung bình, Cao)
6. Máy hút ẩm (Dehumidifier): 3 trạng thái (Tắt, Thấp, Cao)
7. Máy lọc không khí (HEPA Filter): 4 trạng thái (Tắt, Thấp, Trung bình, Cao)

Số lượng tổ hợp hành động ngây thơ (Naive Cartesian Product) là:

$$|\mathcal{A}_{naive}| = 4 \times 3 \times 4 \times 2 \times 4 \times 3 \times 4 = 13.824\text{ hành động}$$

Nếu kết hợp với 720 trạng thái, bảng Q-Table sẽ phình to thành:

$$|\mathcal{S}| \times |\mathcal{A}_{naive}| = 720 \times 13.824 = 9.953.280\text{ ô}$$

Trong học tăng cường dạng bảng, một không gian 10 triệu ô đòi hỏi hàng triệu lượt tương tác để hội tụ, đồng thời phần lớn các tổ hợp ngẫu nhiên sẽ tạo ra các trạng thái **vô nghĩa hoặc nguy hiểm về mặt cơ điện** (ví dụ: vừa bật sưởi vừa bật máy lạnh, hoặc mở cửa sổ giữa trời mưa bão).

---

## 2. Giải pháp: 12 Mẫu Hành động Phối hợp (Coordinated Action Templates)

Hệ thống cô đọng không gian điều khiển thành **12 Mẫu Hành động Phối hợp Chuyên gia**, bao phủ đầy đủ các chế độ vận hành đặc thù của tòa nhà xanh:

| ID | Tên Template | AC Cooling | Heater | Vent Fan | Window | Humidifier | Dehumidifier | HEPA Filter | Ý Nghĩa Điều Khiển Phối Hợp |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `ALL_OFF_STANDBY` | OFF | OFF | OFF (0 m³/h) | CLOSED | OFF | OFF | OFF | Tắt toàn bộ khi phòng không có người |
| **1** | `COMFORT_COOL` | SET_25 (25°C)| OFF | LOW (150 m³/h)| CLOSED | OFF | OFF | LOW (120 m³/h)| Làm mát tiêu chuẩn văn phòng giờ làm việc |
| **2** | `MAX_COOL` | SET_23 (23°C)| OFF | MED (300 m³/h)| CLOSED | OFF | OFF | MED (250 m³/h)| Làm mát cấp tốc khi phòng đông người và nóng |
| **3** | `ECONOMY_COOL` | SET_27 (27°C)| OFF | LOW (150 m³/h)| CLOSED | OFF | OFF | LOW (100 m³/h)| Làm mát tiết kiệm điện trong giờ cao điểm giá điện |
| **4** | `NATURAL_VENTILATION`| OFF | OFF | OFF (0 m³/h) | OPEN (100%) | OFF | OFF | LOW (80 m³/h) | Làm mát tự nhiên qua cửa sổ khi ngoài trời lý tưởng |
| **5** | `ACTIVE_VENT_PURIFY` | OFF | OFF | HIGH (500 m³/h)| CLOSED | OFF | OFF | HIGH (450 m³/h)| Thông gió cưỡng bức xả khí ngạt CO2 và lọc bụi |
| **6** | `COMFORT_HEAT` | OFF | LOW (22°C)| LOW (150 m³/h)| CLOSED | LOW (0.25 kg/h)| OFF | LOW (100 m³/h)| Sưởi ấm tiện nghi mùa lạnh kèm bổ sung ẩm nhẹ |
| **7** | `MAX_HEAT` | OFF | HIGH (24°C)| LOW (150 m³/h)| CLOSED | MED (0.45 kg/h)| OFF | LOW (100 m³/h)| Sưởi ấm tăng cường khi nhiệt độ giảm sâu |
| **8** | `DEHUMIDIFY_ACTIVE` | SET_26 (26°C)| OFF | LOW (150 m³/h)| CLOSED | OFF | HIGH (1.2 kg/h)| LOW (100 m³/h)| Tách ẩm chuyên sâu mùa nồm, duy trì phòng khô ráo |
| **9** | `HUMIDIFY_COMFORT` | SET_25 (25°C)| OFF | LOW (150 m³/h)| CLOSED | HIGH (0.6 kg/h)| OFF | LOW (100 m³/h)| Bổ sung độ ẩm bảo vệ hô hấp trong ngày hanh khô |
| **10**| `ECO_AIR_PURIFY` | OFF | OFF | LOW (100 m³/h)| CLOSED | OFF | OFF | HIGH (450 m³/h)| Lọc không khí HEPA tối đa, không tiêu hao nhiệt |
| **11**| `NIGHT_SETBACK` | SET_26.5 (26.5°C)| OFF | LOW (100 m³/h)| CLOSED | OFF | OFF | LOW (80 m³/h) | Chế độ duy trì ban đêm tiết kiệm năng lượng |

---

## 3. Khóa Liên động Phần cứng và Quy tắc An toàn Cơ điện

Lớp an toàn `MultiDeviceSafetyGateway` hoạt động độc lập và có quyền phủ quyết đơn phương các hành động của Agent:

1. **Khóa liên động Nhiệt (Thermal Interlock):**
   - Tuyệt đối không cho phép máy sưởi và máy lạnh hoạt động đồng thời trong bất kỳ tình huống nào.
2. **Khóa liên động Ẩm (Moisture Interlock):**
   - Cấm đồng thời kích hoạt máy tạo ẩm và máy hút ẩm.
3. **Bộ lọc Điều kiện Môi trường Cửa sổ (Window Environmental Vetoes):**
   - Khi Agent yêu cầu mở cửa sổ (Action 4: `NATURAL_VENTILATION`), Gateway sẽ cưỡng bức đóng cửa sổ và chuyển đổi hành động an toàn nếu:
     - Thời gian đêm muộn (22:00 – 06:00) $\to$ Chuyển sang Action 11 (`NIGHT_SETBACK`).
     - Nhiệt độ ngoài trời $T_{amb} > 33.0^\circ\text{C}$ $\to$ Chuyển sang Action 1 (`COMFORT_COOL`).
     - Độ ẩm ngoài trời $RH_{amb} > 85.0\%$ (nguy cơ mưa gió bão) $\to$ Chuyển sang Action 1 (`COMFORT_COOL`).
     - Bụi mịn ngoài trời $PM2.5_{amb} > 60.0\ \mu\text{g/m}^3$ (sương mù ô nhiễm) $\to$ Chuyển sang Action 10 (`ECO_AIR_PURIFY`).
4. **Bảo vệ Máy nén (Anti-Short-Cycling):**
   - Đảm bảo máy nén làm lạnh không bị bật/tắt liên tục trong vòng 2 bước mô phỏng (tối thiểu 10–20 phút thực tế) để tránh quá dòng khởi động và hỏng hóc cơ khí.
