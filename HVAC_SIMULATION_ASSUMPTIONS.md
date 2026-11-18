# HVAC SIMULATION ASSUMPTIONS & PHYSICAL MODELS

> [!IMPORTANT]
> **Tuyên bố Bản chất Mô phỏng (`RESEARCH_SIMULATION_ONLY`):**  
> Các mô hình toán học trong tài liệu này là các phương trình xấp xỉ vật lý bậc 1 (Grey-Box 2R-1C, Cân bằng ẩm Magnus và Cân bằng nồng độ chất lượng không khí khối tập trung). Hệ thống **không sử dụng mô phỏng động lực học chất lưu (CFD) vi mô** và **không phản ánh sự phân tầng nhiệt độ phức tạp của không gian thực tế**.

---

## 1. Mô hình Truyền nhiệt Nhạy (Sensible Thermal Balance — 2R-1C)

Biến thiên nhiệt độ không khí trong phòng được mô tả bằng phương trình vi phân năng lượng bậc 1:

$$C_{room} \frac{dT_{in}}{dt} = Q_{envelope} + Q_{vent} + Q_{internal} + Q_{heat} - Q_{cool}$$

Trong đó:
- $C_{room} = A_{floor} \cdot c_{specific}$ là tổng nhiệt dung tương đương của phòng (bao gồm không khí, tường bao, sàn bê tông và nội thất). Quy chuẩn giá trị: $c_{specific} \approx 200 - 250\text{ kJ/(m}^2\cdot\text{K)}$.
- $Q_{envelope} = U_{env} \cdot A_{env} \cdot (T_{out} - T_{in})\text{ [W]}$: Nhiệt truyền qua vỏ bao che công trình theo định luật Fourier.
- $Q_{vent} = \frac{\rho_{air} \cdot c_p \cdot \dot{V}_{fresh} \cdot (T_{out} - T_{in})}{3600}\text{ [kW]}$: Enthalpy nhiệt đối lưu trao đổi do thông gió và rò rỉ khí ($\rho_{air} c_p \approx 1.21\text{ kJ/(m}^3\cdot\text{K)}$).
- $Q_{internal}$: Tải nhiệt nội bộ từ người ($q_{occ} \approx 100\text{ W/người}$) và thiết bị điện thoại/máy tính/đèn.
- $Q_{cool}$: Công suất nhiệt làm lạnh hữu ích, tính từ công suất điện tác dụng của máy nén:
  $$Q_{cool} = \begin{cases} 
  0, & \text{khi } T_{in} \le T_{setpoint} \text{ hoặc } AC = \text{OFF} \\
  (P_{elec} - P_{standby}) \cdot COP, & \text{khi máy nén chạy hoạt động}
  \end{cases}$$
- $Q_{heat}$: Công suất nhiệt sưởi hữu ích, $Q_{heat} = P_{heat} \cdot \eta_{heat}$ ($\eta \approx 0.98$).

---

## 2. Mô hình Cân bằng Khối lượng Ẩm Tâm trắc học (Magnus Psychrometric Model)

Độ ẩm không khí được tính toán thông qua tỷ lệ độ ẩm tuyệt đối $W$ (kg hơi nước / kg không khí khô) nhằm đảm bảo bảo toàn khối lượng khi nhiệt độ biến thiên:

1. **Áp suất hơi nước bão hòa (Công thức Magnus-Tetens, sai số < 0.1% trong dải -20°C đến 50°C):**
   $$P_{sat}(T) = 0.61078 \exp\left( \frac{17.27 \cdot T}{T + 237.3} \right)\text{ [kPa]}$$

2. **Chuyển đổi giữa Độ ẩm tương đối ($RH$) và Tỷ lệ ẩm ($W$):**
   $$P_v = \left(\frac{RH}{100}\right) P_{sat}(T)$$
   $$W = 0.62198 \cdot \frac{P_v}{P_{atm} - P_v}\text{ [kg/kg không khí khô]}$$
   $$RH = \frac{W \cdot P_{atm}}{(0.62198 + W) \cdot P_{sat}(T)} \times 100\%$$

3. **Phương trình Vi phân Cân bằng Hơi nước:**
   $$M_{air} \frac{dW_{in}}{dt} = \dot{m}_{occ} + \dot{m}_{humid} + \dot{m}_{air\_flow} - \dot{m}_{dehumid} - \dot{m}_{ac\_latent}$$
   - $\dot{m}_{occ} = N_{occ} \times 0.050\text{ kg/h}$: Hơi nước thoát ra do hô hấp và mồ hôi của người.
   - $\dot{m}_{air\_flow} = \rho_{air} \cdot \dot{V}_{fresh} \cdot (W_{out} - W_{in})$: Hơi ẩm trao đổi qua thông gió tự nhiên và cưỡng bức.
   - $\dot{m}_{ac\_latent}$: Tách ẩm ngưng tụ trên dàn lạnh khi làm mát hoạt động ($0.35\text{ kg/(h}\cdot\text{kW active)}$).
   - $\dot{m}_{humid}, \dot{m}_{dehumid}$: Khối lượng hơi nước bổ sung hoặc tách bỏ của thiết bị chuyên dụng.

---

## 3. Mô hình Chất lượng Không khí Trong nhà (IAQ Mass Balance)

### 3.1. Nồng độ Khí $CO_2$ (ppm)
Phương trình cân bằng bảo toàn khối lượng khí $CO_2$:

$$V_{room} \frac{dC_{co2}}{dt} = G_{co2} \cdot N_{occ} - \dot{V}_{fresh} \cdot (C_{co2, in} - C_{co2, out})$$

- Tốc độ sinh $CO_2$ của người trưởng thành khi làm việc nhẹ: $G_{co2} = 18.000\text{ ppm}\cdot\text{m}^3\text{/h/người}$ ($\approx 0.30\text{ L/phút}$).
- Nồng độ ngoài trời nền: $C_{co2, out} = 415\text{ ppm}$.
- Lưu lượng gió tươi gộp: $\dot{V}_{fresh} = \dot{V}_{infil} + \dot{V}_{vent} + \dot{V}_{window}\text{ [m}^3\text{/h]}$.

### 3.2. Nồng độ Hạt Bụi mịn $PM_{2.5}$ ($\mu\text{g/m}^3$)
$$V_{room} \frac{dC_{pm}}{dt} = G_{pm} \cdot V_{room} + \dot{V}_{unfiltered} \cdot C_{pm, out} - \left(CADR + k_{dep} V_{room} + \dot{V}_{fresh}\right) C_{pm, in}$$

- Tốc độ phát tán hạt bụi nội bộ: $G_{pm} \approx 15\ \mu\text{g/(m}^3\cdot\text{h)}$ khi có người di chuyển.
- Tốc độ phân phối không khí sạch của máy lọc HEPA: $CADR \in [80, 450]\text{ m}^3\text{/h}$.
- Hệ số lắng đọng tự nhiên trên bề mặt sàn/vách: $k_{dep} \approx 0.15\text{ h}^{-1}$.
- Hiệu suất lọc của bộ lọc sơ cấp dàn thông gió (MERV 8-11): $\approx 40\%$ với gió tươi cấp vào.

---

## 4. Hình học Phòng và Thông số Kiến trúc

| Thông số | Phòng Nhỏ (Study/Office) | Phòng Vừa (Open Office) | Phòng Lớn (Conference) |
| :--- | :---: | :---: | :---: |
| **Diện tích sàn ($A_{floor}$)** | $20.0\text{ m}^2$ | $50.0\text{ m}^2$ | $120.0\text{ m}^2$ |
| **Chiều cao trần ($H$)** | $2.6\text{ m}$ | $2.8\text{ m}$ | $3.2\text{ m}$ |
| **Thể tích không khí ($V$)** | $52.0\text{ m}^3$ | $140.0\text{ m}^3$ | $384.0\text{ m}^3$ |
| **Hệ số truyền nhiệt vỏ ($U$)**| $0.65\text{ W/(m}^2\cdot\text{K)}$ | $0.55\text{ W/(m}^2\cdot\text{K)}$ | $0.45\text{ W/(m}^2\cdot\text{K)}$ |
| **Nhiệt dung riêng gộp ($c_{spec}$)**| $200.0\text{ kJ/(m}^2\cdot\text{K)}$ | $220.0\text{ kJ/(m}^2\cdot\text{K)}$ | $250.0\text{ kJ/(m}^2\cdot\text{K)}$ |
| **Tổng nhiệt dung ($C_{room}$)**| $4.000\text{ kJ/K}$ | $11.000\text{ kJ/K}$ | $30.000\text{ kJ/K}$ |
| **Diện tích cửa sổ ($A_{win}$)**| $3.0\text{ m}^2$ | $6.0\text{ m}^2$ | $15.0\text{ m}^2$ |
| **Rò rỉ tự nhiên ($ACH_{base}$)**| $0.25\text{ h}^{-1}$ | $0.20\text{ h}^{-1}$ | $0.18\text{ h}^{-1}$ |
| **Số người thiết kế** | 1 – 2 người | 3 – 4 người | 8 – 10 người |

---

## 5. Bảng Phân loại Minh bạch Nguồn gốc Dữ liệu (Provenance Ledger)

| Trường Dữ liệu | Nhãn Provenance | Giải thích Cơ sở Khoa học |
| :--- | :---: | :--- |
| `ambient_temp_est` | `SIMULATED_ASSUMED` | Ước tính bằng đường cong tuần hoàn ngày đêm theo vĩ độ nhiệt đới Bangkok |
| `ambient_rh_est` | `SIMULATED_ASSUMED` | Đường cong hàm sin nghịch pha với nhiệt độ ngoài trời |
| `ambient_co2_est` | `SIMULATED_ASSUMED` | Nồng độ khí quyển cơ sở toàn cầu ($415\text{ ppm}$) |
| `ambient_pm25_est` | `SIMULATED_ASSUMED` | Biên dạng khói bụi mô phỏng theo kịch bản ($25\ \mu\text{g/m}^3$ hoặc $>60\ \mu\text{g/m}^3$ sương mù) |
| `internal_heat_kw` | `DERIVED` | Tính toán từ tổng công suất đo thực tế của ổ cắm (`plug_kw`) và đèn chiếu sáng (`light_kw`) |
| `temp_c` (indoor) | `SIMULATED` | Nghiệm tích phân số của phương trình cân bằng nhiệt 2R-1C |
| `humidity_rh` (indoor) | `SIMULATED` | Tính từ tỷ lệ ẩm $W$ qua phương trình cân bằng tâm trắc Magnus |
| `co2_ppm` (indoor) | `SIMULATED` | Nghiệm phương trình bảo toàn khối lượng phát thải người và thông gió |
| `pm25_ug_m3` (indoor) | `SIMULATED` | Nghiệm phương trình bảo toàn hạt bụi mịn và tốc độ lọc CADR |
| `powers_kw` | `SIMULATED` | Tính từ đặc tính định mức phụ tải và trạng thái đóng ngắt của 7 nhóm thiết bị |
| `room_geometry` | `SIMULATED_ASSUMED` | Thông số kích thước giả định chuẩn hóa theo tiêu chuẩn thiết kế kiến trúc |
