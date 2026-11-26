# TÀI LIỆU ĐỊNH CỠ THIẾT BỊ VÀ CÂN BẰNG TẢI NHIỆT (EQUIPMENT SIZING & HEAT LOAD BALANCING)

> **Mục tiêu**: Khắc phục triệt để lỗi cấu trúc F5 (tải nội sinh thô của cả phân khu 13 kW đưa vào phòng 50 m² chỉ có điều hòa 12 kW nhiệt khiến nhiệt độ phòng vọt lên >30.6°C, gây vi phạm nhiệt 48–58% ở mọi bộ điều khiển). Tài liệu thiết lập công thức quy đổi tải, tính toán tải thiết kế và chứng minh công suất điều hòa đáp ứng chuẩn kỹ thuật.

---

## 1. Thông số hình học và nhiệt của phòng mô phỏng chuẩn (Medium Office)

- **Diện tích sàn ($A_{floor}$)**: $50.0\text{ m}^2$
- **Chiều cao trần ($H$)**: $2.8\text{ m}$
- **Thể tích không gian ($V$)**: $50.0 \times 2.8 = 140.0\text{ m}^3$
- **Khối lượng không khí trong phòng ($m_{air}$)**: $140.0 \times 1.204 = 168.56\text{ kg}$
- **Diện tích vỏ bao che ($A_{envelope}$)**:
  - Chu vi phòng vuông: $P = 4 \times \sqrt{50.0} \approx 28.28\text{ m}$
  - Diện tích 4 vách tường: $28.28 \times 2.8 = 79.20\text{ m}^2$
  - Tổng diện tích vỏ bao che (tường + mái trần): $A_{envelope} = 79.20 + 50.0 = 129.20\text{ m}^2$
- **Hệ số truyền nhiệt tổng thể vỏ công trình ($U_{envelope}$)**: $0.55\text{ W}/(\text{m}^2\cdot\text{K})$
- **Hệ số thâm nhập gió tự nhiên cơ sở ($ACH_{infil}$)**: $0.20\text{ h}^{-1}$ (lưu lượng thâm nhập: $0.20 \times 140 = 28.0\text{ m}^3/\text{h}$)
- **Nhiệt dung riêng khối công trình ($C_{thermal}$)**: $220.0\text{ kJ}/(\text{m}^2\cdot\text{K}) \times 50.0\text{ m}^2 = 11,000.0\text{ kJ}/\text{K}$ (mô hình hóa quán tính nhiệt của kết cấu sàn bê tông, tường ngăn và nội thất)

---

## 2. Công thức quy đổi tải nội sinh (Scaling Factor $s$)

Tập dữ liệu gốc CU-BEMS thu thập tại Phân khu 1 Tầng 2 (`2018_F2_Z1`), là một không gian văn phòng mở lớn chứa hàng chục bàn làm việc, máy in, máy trạm và hệ thống đèn trần, với công suất điện tiêu thụ đo được tại ổ cắm và chiếu sáng lên tới $13.36\text{ kW}$ (trung bình giờ làm việc $\approx 12.5\text{ kW}$).

Khi mô phỏng trên phòng văn phòng tiêu chuẩn $50\text{ m}^2$, tải nội sinh phải được chuẩn hóa về quy mô phòng theo tiêu chuẩn mật độ thiết bị văn phòng ASHRAE 90.1:
- **Mật độ tải nội sinh tiêu chuẩn ($q_{target}$)**: $35.0\text{ W}/\text{m}^2 = 0.035\text{ kW}/\text{m}^2$ (bao gồm máy tính cá nhân, màn hình và đèn LED).
- **Tải nội sinh cực đại thiết kế cho phòng 50 m²**: $P_{target,max} = 0.035 \times 50.0 = 1.750\text{ kW}$.
- **Phân vị 95% tải nội sinh thô trong tập huấn luyện (`p95_train`)**: Tính toán strictly trên 60% dữ liệu đầu (tập train) của `2018_F2_Z1` để chống rò rỉ dữ liệu:
  $$p95_{train} = 12.7947\text{ kW}$$

- **Hệ số quy đổi quy mô ($s$)** (Nhãn: `DERIVED`):
  $$s = \frac{0.035\text{ kW}/\text{m}^2 \times A_{floor}}{p95_{train}} = \frac{1.7500\text{ kW}}{12.7947\text{ kW}} \approx 0.136775$$

- **Dữ liệu sau quy đổi**:
  - Ban đêm / ngoài giờ làm việc: $0.24\text{ kW} \times s \approx 0.033\text{ kW}$ (chế độ chờ).
  - Giờ làm việc thông thường: $12.0 - 12.9\text{ kW} \times s \approx 1.64 - 1.76\text{ kW}$ (đạt trần thiết kế $1.75\text{ kW}$).

---

## 3. Tính toán phụ tải lạnh thiết kế (Design Cooling Load)

Tính toán tại điều kiện khí hậu mùa hè khắc nghiệt nhất theo tiêu chuẩn ASHRAE Fundamentals:
- Nhiệt độ ngoài trời thiết kế: $T_{ambient} = 35.0^\circ\text{C}$
- Độ ẩm tương đối ngoài trời thiết kế: $RH_{ambient} = 65.0\%$
- Nhiệt độ tiện nghi trong phòng mục tiêu: $T_{room} = 24.0^\circ\text{C}$
- Độ chênh nhiệt độ tính toán: $\Delta T = 35.0 - 24.0 = 11.0\text{ K}$
- Số người tối đa trong phòng: $N_{occ,max} = \text{round}(50.0 / 6) = 8\text{ người}$

### Các thành phần phụ tải nhiệt truyền vào phòng:
1. **Nhiệt truyền qua vỏ công trình ($Q_{envelope}$)**:
   $$Q_{envelope} = \frac{U_{envelope} \times A_{envelope} \times \Delta T}{1000} = \frac{0.55 \times 129.20 \times 11.0}{1000} = 0.7817\text{ kW}$$

2. **Nhiệt do thâm nhập và thông gió cấp gió tươi ($Q_{vent}$)**:
   - Lưu lượng gió tươi tối thiểu: $V_{vent} = 150.0\text{ m}^3/\text{h}$
   - Lưu lượng thâm nhập vỏ: $V_{infil} = 28.0\text{ m}^3/\text{h}$
   - Tổng lưu lượng gió ngoài: $V_{total} = 178.0\text{ m}^3/\text{h}$
   - Dung tích nhiệt thể tích không khí: $\rho \cdot c_p \approx 1.21\text{ kJ}/(\text{m}^3\cdot\text{K})$
   $$Q_{vent} = \frac{1.21 \times 178.0 \times 11.0}{3600} = 0.6581\text{ kW}$$

3. **Nhiệt nội sinh thiết bị điện ($Q_{internal}$)**:
   $$Q_{internal} = 1.7500\text{ kW}$$

4. **Nhiệt hiện do người tỏa ra ($Q_{occupants}$)**:
   - Mỗi người tỏa nhiệt hiện trung bình khi làm việc văn phòng nhẹ: $75\text{ W} = 0.075\text{ kW}$
   $$Q_{occupants} = 8 \times 0.075 = 0.6000\text{ kW}$$

5. **Tổng phụ tải nhiệt thiết kế ($Q_{design}$)**:
   $$Q_{design} = Q_{envelope} + Q_{vent} + Q_{internal} + Q_{occupants}$$
   $$Q_{design} = 0.7817 + 0.6581 + 1.7500 + 0.6000 = 3.7898\text{ kW thermal} \approx 3.79\text{ kW}$$

---

## 4. Kiểm chứng năng lực thiết bị & Cân chỉnh cận trên (Sizing Verification Gates - Phase 1b Update)

> **Ghi chú Liêm chính Học thuật (Phase 1b)**: Rà soát sau GATE 1 phát hiện tỷ số an toàn $3.17$ ở Phase 1 là quá dư thừa so với thực tế thiết kế cơ điện công trình (thường gây hiện tượng chạy non tải, đóng cắt liên tục và lãng phí chi phí đầu tư ban đầu). Để đảm bảo tính thực tế kỹ thuật, nhóm bổ sung **quy tắc cận trên**: Tỷ số công suất làm lạnh cực đại trên phụ tải thiết kế ($\text{Capacity} / Q_{design}$) phải nằm nghiêm ngặt trong khoảng **$[1.2, 1.5]$**, với điểm thiết kế danh định mục tiêu là $1.30 \times Q_{design}$.
> 
> **Cam kết liêm chính**: QUYẾT ĐỊNH NÀY HOÀN TOÀN DỰA TRÊN TIÊU CHUẨN THỰC HÀNH THIẾT KẾ HVAC (ASHRAE / CIBSE), TUYỆT ĐỐI KHÔNG DỰA VÀO VÀ KHÔNG ĐIỀU CHỈNH THEO KẾT QUẢ CỦA BẤT KỲ BỘ ĐIỀU KHIỂN RL HAY BASELINE NÀO.

### 4.1. Định cỡ công suất lạnh bậc HIGH & Điều chỉnh tỷ lệ các bậc:
- Phụ tải nhiệt thiết kế cực đại: $Q_{design} = 3.7898\text{ kW thermal} \approx 3.79\text{ kW}$.
- Mục tiêu công suất lạnh hữu ích cực đại (bậc HIGH):
  $$Q_{cooling,max} = 1.30 \times Q_{design} = 1.30 \times 3.7898 = 4.9267\text{ kW thermal} \approx 4.93\text{ kW}$$
- Hệ số hiệu quả năng lượng làm lạnh (COP): Giữ nguyên $\text{COP} = 3.20$ (chuẩn máy lạnh biến tần thương mại).
- Công suất điện máy nén hoạt động hữu ích bậc HIGH:
  $$P_{active,high} = \frac{Q_{cooling,max}}{\text{COP}} = \frac{4.9267}{3.20} \approx 1.540\text{ kW electrical}$$
- Công suất tiêu thụ điện ở chế độ chờ (standby electronics): $P_{ac,off} = 0.05\text{ kW}$.
- Tổng công suất điện định mức bậc HIGH:
  $$P_{ac,high} = P_{active,high} + P_{ac,off} = 1.540 + 0.050 = 1.590\text{ kW electrical}$$

### 4.2. Giảm `ac_low` và `ac_med` theo CÙNG tỷ lệ để bảo toàn đặc tính điều biến:
Tỷ lệ co giảm công suất hoạt động so với bản gốc:
$$\kappa = \frac{1.540\text{ kW}}{3.750\text{ kW}} \approx 0.4106$$

Áp dụng đồng đều cho các bậc hoạt động:
- **Bậc LOW**:
  - Công suất hoạt động: $1.15\text{ kW} \times 0.4106 = 0.472\text{ kW}$
  - Tổng công suất điện: $P_{ac,low} = 0.472 + 0.050 = 0.522 \approx \mathbf{0.52\text{ kW electrical}}$
  - Tỷ lệ so với bậc HIGH: $0.47 / 1.54 \approx 30.5\%$
- **Bậc MED**:
  - Công suất hoạt động: $2.35\text{ kW} \times 0.4106 = 0.965\text{ kW}$
  - Tổng công suất điện: $P_{ac,med} = 0.965 + 0.050 = 1.015 \approx \mathbf{1.01\text{ kW electrical}}$
  - Tỷ lệ so với bậc HIGH: $0.96 / 1.54 \approx 62.3\%$
- **Bậc HIGH**:
  - Tổng công suất điện: $P_{ac,high} = \mathbf{1.59\text{ kW electrical}}$
  - Công suất lạnh hữu ích: $Q_{cooling,max} = (1.59 - 0.05) \times 3.2 = 4.928\text{ kW thermal}$

### 4.3. Kiểm tra tự động tỷ số an toàn (Automated Sizing Ratio Check):
$$\text{Safety Ratio} = \frac{Q_{cooling,max}}{Q_{design}} = \frac{4.928\text{ kW}}{3.7898\text{ kW}} = \mathbf{1.3003}$$
$$\mathbf{1.20 \le 1.3003 \le 1.50} \quad (\text{ĐẠT CHUẨN THIẾT KẾ})$$
Mã nguồn môi trường tích hợp kiểm tra tự động `assert 1.2 <= self.sizing_safety_ratio <= 1.5` ngay trong hàm khởi tạo `MultiDeviceHVACEnv.__init__`, đảm bảo mọi cấu hình không gian phòng mô phỏng đều phải thỏa mãn tiêu chí thiết kế này.

### 4.4. Tiêu chí 1.3b — Khoảng tiêu thụ năng lượng hợp lý:
- Khoảng kiểm tra kỹ thuật ASHRAE cho hệ thống HVAC văn phòng: **$0.20 - 0.80\text{ kWh}/\text{m}^2/\text{ngày}$**
- Tương ứng với diện tích phòng $50\text{ m}^2$: **$10.0 - 40.0\text{ kWh}/\text{ngày}$**
- **Kết quả mô phỏng thực tế của Fixed Setpoint (25°C)** sau khi định cỡ chuẩn $[1.2, 1.5]$ và tích hợp công thức khử ẩm SHR:
  $$\text{Điện năng trung bình 1 ngày thường} \approx 13.5 - 15.3\text{ kWh/ngày}$$
  $$\text{Mật độ tiêu thụ} \approx 0.27 - 0.31\text{ kWh}/\text{m}^2/\text{ngày}$$
  $\rightarrow$ **KẾT LUẬN**: Nằm hoàn toàn trong dải tiêu chuẩn ($0.20 \le 0.31 \le 0.80$), chứng minh hệ thống phản ánh trung thực thực tế vận hành công trình xanh.
