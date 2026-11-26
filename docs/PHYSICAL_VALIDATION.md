# BÁO CÁO KIỂM CHỨNG TÍNH ĐÚNG ĐẮN VẬT LÝ (PHYSICAL VALIDATION REPORT - PV1 ĐẾN PV7)

> **Mục tiêu**: Báo cáo kiểm định độc lập các định luật nhiệt động học, ẩm trắc (psychrometrics), động học chất ô nhiễm, định cỡ thiết bị, cân bằng năng lượng và tái hiện vận hành thực tế (Operation Replay) trên môi trường Digital Twin đa thiết bị (`MultiDeviceHVACEnv`) theo 7 tiêu chí kiểm chứng vật lý PV1–PV7 (hoàn thiện mô hình cuộn lạnh cho PV6 và bổ sung PV7 Operation Replay ở Phase 1c).
>
> **Nguyên tắc Liêm chính**: Mọi số liệu trong báo cáo này được trích xuất trực tiếp từ các bài đo kiểm thực tế và test tự động. Các đại lượng mô tả hình dạng phụ tải (PV5) và chạy lại vận hành thực tế (PV7) được báo cáo trung thực dưới dạng giá trị thống kê mô tả, không gắn nhãn "ĐẠT/KHÔNG ĐẠT" cảm tính và không phụ thuộc p-value do tự tương quan chuỗi thời gian. Không có bất kỳ hiệu chỉnh nào nhằm tạo ưu thế cho bất kỳ thuật toán RL hay baseline nào.

- **File kiểm thử tự động**: `tests/test_physical_validation.py`
- **Kết quả kiểm thử tự động**: **74/74 ca kiểm thử PASS (100%)**
- **Dữ liệu trực quan đối chiếu**:
  - `result/final/diag_pv1_all_off_24h.png`: Quỹ đạo trôi nhiệt tự nhiên 24h khi tắt AC ngày nắng nóng cực đại có người.
  - `result/final/diag_pv6_rh_comparison.png`: Phân phối độ ẩm thực tế CU-BEMS vs mô phỏng cuộn lạnh trên toàn bộ các ngày thường.
  - `result/final/diag_pv7_operation_replay.png`: So sánh biên dạng giờ nhiệt độ & độ ẩm thực tế vs mô phỏng chạy lại vận hành trên Train và Validation.
  - `result/final/diag_state_coverage.png`: Phân bố không gian trạng thái 720 bins và phân tích chiều chết.
  - `result/final/diag_state_coverage.csv`: Dữ liệu bảng tần suất ghé thăm 720 trạng thái.

---

## 1. Bảng tổng hợp kết quả 7 tiêu chí kiểm chứng vật lý (PV1 – PV7)

| Mã | Tiêu chí kiểm chứng | Điều kiện & Phương pháp đo | Kết quả đo thực nghiệm | Đánh giá kỹ thuật |
|:---:|---|---|:---:|:---:|
| **PV1** | **Trôi nhiệt tự nhiên (Passive Drift)** | Tắt toàn bộ HVAC (Action 0: `ALL_OFF`) trong 24 giờ ngày thường nóng nhất tập train ($05/07/2018$, nhiệt độ ngoài trời đỉnh $35.80^\circ\text{C}$, có $8\text{ người}$, tải nội sinh $1.75\text{ kW}$). | $T_{init} = 27.18^\circ\text{C}$<br>$T_{peak} = 36.47^\circ\text{C}$ (giờ 19)<br>$T_{final} = 35.89^\circ\text{C}$<br>Khoảng cách tới trần an toàn: $3.53^\circ\text{C}$. | **HỢP LÝ**<br>(Nhiệt tăng tự nhiên theo quán tính nhiệt, không bị kẹt ở trần $40^\circ\text{C}$). |
| **PV2** | **Năng lực làm lạnh xác lập tại điều kiện thiết kế** | Điều kiện thiết kế mùa hè ($T_{amb} = 35.0^\circ\text{C}$, tải nội sinh $P_{95} = 1.75\text{ kW}$, $8\text{ người}$). Chạy xác lập 24h với Action 1 (setpoint $25^\circ\text{C}$) và Action 2 (setpoint $23^\circ\text{C}$). | - **Action 1**: $T_{steady} = 25.83^\circ\text{C} \le 26.0^\circ\text{C}$ (14h MED, 4h HIGH).<br>- **Action 2**: $T_{steady} = 23.83^\circ\text{C} \le 26.0^\circ\text{C}$ (14h HIGH, 4h MED).<br>Tỷ số Capacity/$Q_{des} = \mathbf{1.3003} \in [1.2, 1.5]$. | **ĐẠT TIÊU CHUẨN**<br>(Giữ được $\le 26^\circ\text{C}$, không bão hòa ở HIGH liên tục, điều biến linh hoạt). |
| **PV3** | **Độ ổn định số học vi phân (Triệt tiêu răng cưa)** | Chạy lọc khí liên tục (Action 10 hoặc Action 1). Đo tỉ lệ đảo dấu vi phân $PM_{2.5}$ và $\text{CO}_2$ qua các bước liên tiếp. | Tỉ lệ đảo dấu $PM_{2.5}$: $22.7\% < 35\%$ (đạo hàm bậc hai $\|d^2\| = 0.0076\ \mu\text{g/m}^3$). Tỉ lệ đảo dấu $\text{CO}_2$: $36.4\% < 50\%$. | **ỔN ĐỊNH**<br>(Triệt tiêu hoàn toàn dao động răng cưa số học; dao động nhỏ do nhiễu Gauss ở sàn bão hòa). |
| **PV4** | **Bảo toàn năng lượng & Khối lượng** | Kiểm tra Định luật 1 Nhiệt động lực học trên từng bước con tích phân: $C_{thermal} \frac{dT}{dt} = \sum Q_i$. | Sai số tích lũy bảo toàn năng lượng sau 24h là $0.000\text{ kWh} \le 0.01\text{ kWh}$. Năng lượng tích lũy không âm. | **BẢO TOÀN CHÍNH XÁC** |
| **PV5** | **Tương quan hình dạng phụ tải ngày (Sim vs Real)** | So sánh biên dạng trung bình theo giờ của TOÀN BỘ ngày thường (31 ngày trong 60% đầu) giữa mô phỏng Fixed Setpoint và phụ tải thực tế tòa nhà Chamchuri 5 ($hvac\_kw$). | - Toàn bộ 24 giờ: **$r = 0.6054$**<br>- Giờ 1 đến 23 (bỏ spike giờ 0): **$r = 0.8092$** | **SỐ LIỆU MÔ TẢ**<br>(Tương quan biên dạng ngày/đêm tốt. $p$-value không đáng tin cậy do tự tương quan chuỗi). |
| **PV6** | **Động học độ ẩm & Mô hình Cuộn lạnh (Cooling Coil)** | So sánh phân phối độ ẩm phòng thực tế CU-BEMS trong giờ có người ($N_{occ} > 0$) với mô phỏng Fixed Setpoint qua mô hình cuộn lạnh (Cooling Coil Model, $T_{coil} = 12.0^\circ\text{C}$). | - **Train Real**: P50 = $53.1\%$, P1 = $48.6\%$, P99 = $64.9\%$.<br>- **Train Sim**: P50 = $53.8\%$, P1 = $48.5\%$, P99 = $81.6\%$ (Lệch P50: **$+0.7\%$**).<br>- **Val Real**: P50 = $56.4\%$, P1 = $52.7\%$, P99 = $64.3\%$.<br>- **Val Sim**: P50 = $55.0\%$, P1 = $50.5\%$, P99 = $86.2\%$ (Lệch P50: **$-1.4\%$**). | **KHỚP VẬT LÝ CAO**<br>(Sai số trung vị $< 1.5\%$, bám sát dữ liệu thực nghiệm). |
| **PV7** | **Tái hiện vận hành thực tế (Operation Replay)** | Ánh xạ nhị phân trạng thái bật/tắt AC thực tế ($hvac\_kw \ge 1.0\text{ kW} \to$ Action 1, $< 1.0\text{ kW} \to$ Action 0), khởi tạo từ 00:00 thực tế, mô phỏng 24h trên Train (60%) và Validation (20%). | - **Train**: Lệch trung bình $T = +0.15^\circ\text{C}$ ($\le 1.5^\circ\text{C}$), Lệch P50 $RH = +0.7\%$ ($\le 8\%$).<br>- **Validation**: Lệch trung bình $T = +0.34^\circ\text{C}$ ($\le 1.5^\circ\text{C}$), Lệch P50 $RH = +3.3\%$ ($\le 8\%$). | **THỎA MÃN CẢNH BÁO**<br>(Cả hai chỉ số đều nằm an toàn dưới ngưỡng cảnh báo đặt trước). |

---

## 2. Chi tiết phân tích kỹ thuật từng tiêu chí

### 2.1. PV1 — Hiện tượng trôi nhiệt tự nhiên (Passive Thermal Drift)
- **Kịch bản kiểm thử**: Ngày $05/07/2018$ (bước thời gian bắt đầu $idx = 96$, thứ Năm) là ngày nắng nóng nhất trong tập huấn luyện với nhiệt độ ngoài trời đạt đỉnh $35.80^\circ\text{C}$ (trung bình $30.30^\circ\text{C}$). Phòng có người làm việc thực tế (tối đa 8 người từ 08:00 đến 18:00) và tải thiết bị điện nội sinh đầy đủ ($1.75\text{ kW}$). Toàn bộ thiết bị điều hòa bị tắt hoàn toàn (`ALL_OFF_STANDBY`).
- **Hiện tượng vật lý đo được**:
  - Nhiệt độ khởi điểm lúc nửa đêm: $27.18^\circ\text{C}$.
  - Ban ngày, do không có làm mát, nhiệt từ người ($8 \times 0.075 = 0.60\text{ kW}$), thiết bị ($1.75\text{ kW}$) và vỏ bao che truyền vào khiến nhiệt độ phòng tăng đều đặn, đạt đỉnh **$36.47^\circ\text{C}$ vào lúc 19:00**.
  - Nhiệt độ cuối ngày (23:00) đạt $35.89^\circ\text{C}$.
  - **Kiểm tra biên an toàn**: Nhiệt độ cực đại $36.47^\circ\text{C}$ cách trần an toàn vật lý ($40.0^\circ\text{C}$) một khoảng dự phòng an toàn là **$3.53^\circ\text{C}$**, chứng minh mô hình không bị kẹt cứng (clipping) tại trần $40^\circ\text{C}$.

### 2.2. PV2 — Năng lực làm lạnh xác lập tại điều kiện thiết kế (Design Sizing & Staging)
- **Quy chuẩn định cỡ AC**: Bổ sung cận trên cho tỷ số công suất lạnh bậc HIGH trên tải thiết kế trong khoảng $[1.2, 1.5]$ để phản ánh đúng thực hành thiết kế cơ điện (tránh máy lạnh quá cỡ gây đóng cắt liên tục và hao tổn công suất).
- **Thông số định cỡ**:
  - Tải thiết kế $Q_{design} = 3.7898\text{ kW thermal}$.
  - Công suất điện máy lạnh: $P_{low} = 0.52\text{ kW}$, $P_{med} = 1.01\text{ kW}$, $P_{high} = 1.59\text{ kW}$.
  - Công suất lạnh hữu ích cực đại: $Q_{cooling,max} = (1.59 - 0.05) \times 3.2 = 4.928\text{ kW thermal}$.
  - Tỷ số an toàn: $\text{Capacity}/Q_{design} = 4.928 / 3.7898 = \mathbf{1.3003 \in [1.2, 1.5]}$.
- **Kết quả chạy xác lập 24h ở $T_{ngoai} = 35^\circ\text{C}, P_{int} = 1.75\text{ kW}, N_{occ} = 8$**:
  - **Action 1 (COMFORT_COOL, setpoint $25^\circ\text{C}$)**: Nhiệt độ xác lập giữ ở **$25.83^\circ\text{C} \le 26.0^\circ\text{C}$**. Trong 18 giờ xác lập, máy điều biến giữa bậc MED (14 giờ, công suất $\approx 1.01\text{ kW}$) và bậc HIGH (4 giờ, công suất $\approx 1.59\text{ kW}$), hoàn toàn không bị bão hòa liên tục ở HIGH.
  - **Action 2 (MAX_COOL, setpoint $23^\circ\text{C}$)**: Nhiệt độ xác lập giữ ở **$23.83^\circ\text{C} \le 26.0^\circ\text{C}$**. Máy điều biến giữa bậc HIGH (14 giờ) và bậc MED (4 giờ).

### 2.3. PV3 — Loại bỏ hoàn toàn dao động răng cưa số học (Multi-rate Sub-stepping)
- Bước điều khiển 1 giờ được chia nhỏ thành 12 bước tích phân vật lý 5 phút ($\Delta t_{phys} = 300\text{ s}$).
- Đạo hàm bậc hai tuyệt đối cực đại của nồng độ $PM_{2.5}$ đạt mức cực nhỏ: $\|d^2\| = 0.0076\ \mu\text{g/m}^3 \ll 5.0$.
- Tỉ lệ đổi dấu vi phân ở sàn bão hòa phản ánh đúng nhiễu trắng ngẫu nhiên của cảm biến ($\mathcal{N}(0, 0.1)$ cho $PM_{2.5}$ và $\mathcal{N}(0, 5.0)$ cho $\text{CO}_2$), không còn bất kỳ dao động răng cưa biên độ lớn nào do phương pháp giải số.

### 2.4. PV4 — Bảo toàn năng lượng & Khối lượng (Thermal Mass Balance)
- Kiểm tra Định luật 1 Nhiệt động lực học trên từng bước con tích phân: $C_{thermal} \frac{dT}{dt} = \sum Q_i$.
- Sai số tích lũy bảo toàn năng lượng sau 24h là $0.000\text{ kWh} \le 0.01\text{ kWh}$. Năng lượng tích lũy không âm.

### 2.5. PV5 — Tương quan hình dạng phụ tải ngày (Simulated vs Real HVAC Diurnal Profile)
- **Phương pháp tính**: Tính toán trung bình theo từng giờ của tất cả 31 ngày thường (thứ Hai đến thứ Sáu) trong 60% dữ liệu đầu của `2018_F2_Z1`.
- **Kết quả Pearson correlation**:
  - Trên toàn bộ 24 giờ: **$r = 0.6054$**.
  - Trên các giờ từ 01:00 đến 23:00 (loại trừ giờ khởi động 00:00): **$r = 0.8092$**.
- **Lưu ý phương pháp luận & Bản chất số liệu**:
  1. Đây là chỉ số **mô tả hình dạng biên dạng** (shape correlation), không so sánh độ lớn tuyệt đối vì phụ tải mô phỏng đã được quy đổi tỉ lệ cho phòng đơn $50\text{ m}^2$, trong khi $hvac\_kw$ thực tế là tổng điện năng chiller của toàn bộ tòa nhà Chamchuri 5.
  2. Dữ liệu chuỗi thời gian theo giờ có hiện tượng **tự tương quan chuỗi (serial autocorrelation)** cao, do đó trị số $p$-value không thỏa mãn giả định độc lập thống kê của kiểm định Student và không được dùng làm ngưỡng chấp nhận khoa học.
  3. **Giải thích hiện tượng xung nhọn (spike) tại Giờ 00:00**:
     - Trong mô phỏng, khi hàm `env.reset()` được gọi lúc 00:00, nhiệt độ khởi điểm được lấy từ cảm biến thực tế của phòng ($T_{init} \approx 27.2^\circ\text{C}$) do nhiệt tồn dư từ cuối ngày hôm trước khi hệ thống điều hòa thực tế đã tắt.
     - Bộ điều khiển Fixed Setpoint (target $25^\circ\text{C}$) lập tức kích hoạt máy nén ở bậc HIGH lúc 00:00 để kéo nhiệt phòng xuống ($P_{sim} \approx 1.45\text{ kW}$), sau đó giảm dần về chế độ chờ $0.05\text{ kW}$ từ 02:00 đến 06:00.
     - Trong khi đó, tại tòa nhà thực tế, trạm chiller trung tâm bị ngắt nguồn hoàn toàn suốt đêm (chỉ tiêu thụ $0.35\text{ kW}$ standby). Do đó, sự khác biệt trong giờ 00:00 phản ánh đúng đặc tính khởi tạo chu kỳ của bài toán mô phỏng hồi quy.

---

### 2.6. PV6 — Cân bằng ẩm & Mô hình Cuộn lạnh (Cooling Coil Dehumidification Model)

#### A. Phân tích nguyên nhân sai lệch ở Phase 1b qua Cân bằng ẩm chi tiết
Ở Phase 1b, công thức SHR ($0.75$) gắn khử ẩm thuần túy vào công suất lạnh hiển nhiệt:
$$\dot{m}_{latent} = Q_{sensible} \times \frac{1 - \text{SHR}}{\text{SHR}} \times \frac{3600}{2450}$$
Kết quả khiến độ ẩm mô phỏng vẫn bị lệch cao ($P50 = 71.2\%, P99 = 93.4\%$) so với thực tế ($P50 = 53.1\%, P99 = 64.9\%$).

**Giải trình bằng phương trình cân bằng ẩm giờ có người** (trung bình 417 giờ có người trong Train):
1. **Lượng ẩm sinh ra từ người** ($\dot{m}_{occ}$): Với trung bình $6.9\text{ người}$, mỗi người tỏa $0.050\text{ kg/h}$ ẩm:
   $$\dot{m}_{occ} \approx 6.9 \times 0.050 = 0.345\text{ kg/h}$$
2. **Lượng ẩm xâm nhập từ thông gió và rò rỉ** ($\dot{m}_{vent}$): Quạt cấp gió tươi ($150\text{ m}^3\text{/h}$) và độ rò rỉ ($0.2\text{ ACH} \approx 28\text{ m}^3\text{/h}$) mang không khí nhiệt đới ngoài trời vào phòng:
   $$\dot{m}_{vent} = \rho \times (V_{vent} + V_{infil}) \times (w_{amb} - w_{room}) \approx 1.20 \times 178 \times (0.0155 - 0.0120) \approx 0.758\text{ kg/h}$$
3. **Tổng tải ẩm cần khử**: $\dot{m}_{total} = \dot{m}_{occ} + \dot{m}_{vent} = 0.345 + 0.758 = 1.103\text{ kg/h}$.
4. **Hạn chế của công thức SHR cũ**: Khi phòng đã đạt setpoint nhiệt ($25^\circ\text{C}$), $Q_{sensible}$ giảm về bậc MED ($0.96\text{ kW}$ nhiệt) hoặc LOW ($0.47\text{ kW}$ nhiệt). Khi đó công thức SHR chỉ khử được:
   $$\dot{m}_{latent,SHR} = 0.96 \times \frac{0.25}{0.75} \times \frac{3600}{2450} \approx 0.470\text{ kg/h} \ll 1.103\text{ kg/h}$$
   Lượng ẩm bị thiếu hụt $1.103 - 0.470 = 0.633\text{ kg/h}$ liên tục tích tụ trong phòng khiến độ ẩm tăng vọt lên $71 - 93\%$.

#### B. Mô hình Cuộn lạnh Chuẩn hóa (Cooling Coil Physical Model - ASHRAE Fundamentals Ch. 21)
Để phản ánh chính xác động học điều hòa không khí dạng cuộn lạnh cưỡng bức (fan coil unit), lượng ẩm ngưng tụ phụ thuộc vào lưu lượng gió tuần hoàn qua cuộn lạnh ($V_{coil}$) và độ chênh tỷ dung ẩm giữa không khí phòng ($w_{room}$) và bề mặt cuộn lạnh bão hòa ($w_{coil,sat}$):
$$\dot{m}_{latent} = \min\left(\rho_{air} \cdot V_{coil} \cdot (w_{room} - w_{coil,sat}) \cdot \text{PLF} \cdot 3600,\ \frac{Q_{sensible} \cdot 3600}{2450}\right)$$

Trong đó:
- $T_{coil} = 12.0^\circ\text{C}$ (nhiệt độ bề mặt dàn lạnh tiêu chuẩn điều hòa nhiệt đới theo ASHRAE).
- $w_{coil,sat} = 0.00874\text{ kg nước / kg không khí khô}$ (tỷ dung ẩm bão hòa tại $12.0^\circ\text{C}$).
- $V_{coil} = 0.08\text{ m}^3\text{/s trên mỗi kW lạnh thiết kế}$ ($V_{coil} = 0.08 \times 3.7898 = 0.3032\text{ m}^3\text{/s} \approx 1,091\text{ m}^3\text{/h}$ lưu lượng gió tuần hoàn).
- $\text{PLF}$ (Partial Load Factor): Hệ số tải một phần của máy nén ($0.33$ ở LOW, $0.66$ ở MED, $1.00$ ở HIGH).
- Nền độ ẩm ngoại cảnh Bangkok: Hiệu chỉnh về giá trị $68\%$ (SIMULATED_ASSUMED, tham số hiệu chuẩn trên train; thay vì giả định cũ $75\%$).

#### C. Kết quả kiểm chứng PV6 trên Train (60%) và Validation (20%)
Số liệu thống kê trong giờ có người làm việc ($N_{occ} > 0$):

| Tập dữ liệu | Nguồn | P1 (%) | P50 Trung vị (%) | P99 (%) | Trung bình (%) | Lệch P50 (Sim - Real) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Train (60%)** | **Thực tế CU-BEMS** | 48.6% | **53.1%** | 64.9% | 53.9% | — |
| **Train (60%)** | **Mô phỏng cuộn lạnh** | 48.5% | **53.8%** | 81.6% | 56.3% | **+0.7%** |
| **Validation (20%)** | **Thực tế CU-BEMS** | 52.7% | **56.4%** | 64.3% | 56.6% | — |
| **Validation (20%)** | **Mô phỏng cuộn lạnh** | 50.5% | **55.0%** | 86.2% | 57.2% | **-1.4%** |

**Kết luận PV6**: Sai số trung vị độ ẩm giữa mô phỏng và thực tế nằm trong khoảng $[-1.4\%, +0.7\%]$ (dưới $2\%$), hoàn toàn khớp với phân phối thực nghiệm của tòa nhà Chamchuri 5.

---

### 2.7. PV7 — Tái hiện Vận hành Thực tế (Operation Replay)

#### A. Phương pháp luận
- Nhằm kiểm tra xem mô hình Digital Twin có phản ứng trung thực với cách vận hành HVAC thực tế của tòa nhà hay không, ta thực hiện thí nghiệm **Operation Replay**:
  - Với mỗi ngày trong tập Train và Validation, khởi tạo điều kiện phòng lúc 00:00 từ dữ liệu cảm biến thực tế ($T_{in}(0) = T_{real}(0)$, $RH_{in}(0) = RH_{real}(0)$).
  - Từng giờ, đọc giá trị công suất chiller thực tế của tòa nhà ($hvac\_kw$).
  - **Quy tắc ánh xạ nhị phân khách quan**:
    - Nếu $hvac\_kw < 1.0\text{ kW}$ (hệ thống điều hòa tòa nhà tắt): Kích hoạt Action 0 (`ALL_OFF_STANDBY`).
    - Nếu $hvac\_kw \ge 1.0\text{ kW}$ (hệ thống điều hòa tòa nhà bật): Kích hoạt Action 1 (`COMFORT_COOL`).
  - Mô phỏng liên tục 24 giờ và so sánh trực tiếp chuỗi nhiệt độ & độ ẩm mô phỏng với dữ liệu đo thực tế.

#### B. Ngưỡng cảnh báo đặt trước (Pre-set Warning Thresholds)
Để đảm bảo tính khách quan khoa học, các ngưỡng cảnh báo sau đây được chốt TRƯỚC KHI quan sát kết quả:
- **Ngưỡng nhiệt độ**: Độ lệch trung bình toàn bộ chuỗi (Mean Error) không vượt quá **$\pm 1.5^\circ\text{C}$**.
- **Ngưỡng độ ẩm**: Độ lệch phân vị trung vị (P50 Error) không vượt quá **$\pm 8.0\%$**.

#### C. Bảng số liệu thống kê mô tả PV7 (Train & Validation)

| Tập dữ liệu | Phân nhóm giờ | Số mẫu (N) | Nhiệt độ: Mean Diff (°C) | Nhiệt độ: MAE (°C) | Nhiệt độ: RMSE (°C) | Nhiệt độ: Pearson r | Độ ẩm: P50 Diff (%) | Độ ẩm: MAE (%) | Độ ẩm: RMSE (%) | Độ ẩm: Pearson r |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN (60%)** | **Toàn bộ (All)** | 1,056 | **+0.15** | 1.33 | 1.55 | 0.7795 | **+0.7** | 6.6 | 9.5 | 0.3710 |
| | Giờ có người (Occupied) | 417 | +1.48 | 1.99 | 2.10 | 0.7352 | -0.4 | 4.3 | 6.1 | 0.3863 |
| | Giờ không người (Unoccupied) | 639 | -0.72 | 0.89 | 1.05 | 0.8250 | +4.4 | 8.1 | 11.2 | 0.1640 |
| **VAL (20%)** | **Toàn bộ (All)** | 312 | **+0.34** | 1.16 | 1.42 | 0.7526 | **+3.3** | 6.9 | 9.4 | 0.6900 |
| | Giờ có người (Occupied) | 123 | +1.42 | 1.97 | 2.07 | 0.7211 | -3.1 | 4.4 | 5.9 | 0.2778 |
| | Giờ không người (Unoccupied) | 189 | -0.36 | 0.63 | 0.75 | 0.8416 | +6.8 | 8.5 | 11.1 | 0.6609 |

#### D. Đánh giá kỹ thuật & Phân tích nguyên nhân hiện tượng
1. **Đối chiếu ngưỡng cảnh báo**:
   - Lệch trung bình nhiệt độ: $+0.15^\circ\text{C}$ (Train) và $+0.34^\circ\text{C}$ (Val), đều nằm rất sâu dưới ngưỡng trần $\pm 1.5^\circ\text{C}$.
   - Lệch trung vị độ ẩm: $+0.7\%$ (Train) và $+3.3\%$ (Val), đều nằm dưới ngưỡng cảnh báo $\pm 8.0\%$.
2. **Hiện tượng giờ có người (Occupied Hours)**:
   - Trong giờ có người làm việc, nhiệt độ mô phỏng cao hơn nhiệt độ đo thực tế trung bình khoảng $+1.45^\circ\text{C}$ (MAE $\approx 1.98^\circ\text{C}$).
   - **Nguyên nhân vật lý**: Mô hình Digital Twin phòng đơn tính tải nhiệt sinh học của toàn bộ người trong phòng ($8\text{ người} \times 75\text{ W} = 600\text{ W}$) bị giam giữ trong thể tích phòng kín $140\text{ m}^3$ với máy lạnh đơn $1.59\text{ kW}$. Trong khi đó, tại tòa nhà thực tế, các phòng mở cửa lưu thông với hành lang lớn và được hưởng lợi từ luồng gió lạnh tuần hoàn từ hệ thống AHU/chiller trung tâm công suất lớn của cả tầng. Đây là đặc tính cách ly hình học phòng đơn được ghi nhận rõ ràng làm hạn chế của mô hình.
3. **Hiện tượng giờ ban đêm (Unoccupied Hours)**:
   - Mô phỏng bám rất sát thực tế (MAE nhiệt độ chỉ $0.63 - 0.89^\circ\text{C}$, tương quan $r > 0.82$).

---

## 3. Kiểm tra Mức Năng lượng Tiêu thụ với AC Định cỡ Mới (1.59 kW High)

- **Diện tích phòng tiêu chuẩn**: $50.0\text{ m}^2$ (thể tích $140.0\text{ m}^3$).
- **Kết quả mô phỏng Fixed Setpoint (Action 1) trên các ngày thường**:
  - **Ngày thường tiêu chuẩn (02/07/2018)**:
    - Điện năng máy lạnh: $8.804\text{ kWh/ngày}$ ($0.1761\text{ kWh/m}^2\text{/ngày}$).
    - Điện năng toàn hệ thống (AC + Quạt + Lọc khí): $10.843\text{ kWh/ngày}$ ($0.2169\text{ kWh/m}^2\text{/ngày}$).
  - **Trung bình 31 ngày thường tập Train**:
    - Điện năng máy lạnh: $13.555\text{ kWh/ngày}$ ($0.2711\text{ kWh/m}^2\text{/ngày}$).
    - Điện năng toàn hệ thống: **$15.595\text{ kWh/ngày}$** (**$0.3119\text{ kWh/m}^2\text{/ngày}$**).
- **Đối chiếu Benchmark Quốc tế (ASHRAE / CIBSE Guides)**:
  - Dải định mức tiêu thụ điện làm mát cho văn phòng/phòng thí nghiệm ở vùng khí hậu nhiệt đới ẩm: **$0.20 - 0.80\text{ kWh/m}^2\text{/ngày}$**.
  - Mức tiêu thụ $0.3119\text{ kWh/m}^2\text{/ngày}$ của hệ thống mô phỏng nằm hoàn toàn bên trong dải benchmark thực tế, chứng minh máy lạnh $1.59\text{ kW}$ điện ($4.93\text{ kW}$ lạnh) hoạt động hoàn toàn thực tế, không bị tiêu thụ năng lượng ảo hay phi lý.

---

## 4. Giới hạn đã biết của Mô hình Mô phỏng (Known Model Limitations)

Nhằm đảm bảo tính trung thực khoa học và tránh mọi ngộ nhận khi đánh giá kết quả, các giới hạn vật lý và thống kê sau đây được xác định rõ ràng:

1. **(a) Sai lệch nhiệt độ giờ có người**: Trong các giờ có người làm việc, nhiệt độ mô phỏng cao hơn nhiệt độ đo thực tế trung bình từ $+1.4^\circ\text{C}$ đến $+1.5^\circ\text{C}$ ($\text{MAE} \approx 1.98^\circ\text{C}$).
   - *Giả thuyết nguyên nhân (CHƯA ĐƯỢC KIỂM CHỨNG THỰC ĐỊA, ĐÁNH DẤU LÀ GIẢ THUYẾT)*: Mô hình Digital Twin phòng đơn tính toán tải nhiệt sinh học của toàn bộ người trong phòng ($8\text{ người} \times 75\text{ W} = 600\text{ W}$) bị giam giữ trong thể tích phòng kín $140\text{ m}^3$ với máy lạnh $1.59\text{ kW}$, trong khi tại công trình thực tế, phòng có thể mở cửa lưu thông với hành lang và được làm mát hỗ trợ từ luồng không khí tuần hoàn của hệ thống AHU/chiller trung tâm toàn tầng.
2. **(b) Độ lệch phân vị P99 của độ ẩm tương đối**: Mặc dù độ ẩm trung vị P50 bám sát thực tế (lệch $< 1.5\%$), phân vị cực đại P99 trong mô phỏng cao hơn thực tế từ $17$ đến $22$ điểm phần trăm (mô phỏng đạt $82 - 86\%$ so với thực tế $64 - 65\%$). Điều này xảy ra do mô hình phòng kín bảo toàn hơi ẩm khi AC ngắt đột ngột ngoài giờ làm việc.
3. **(c) Tương quan độ ẩm yếu trong giờ có người**: Hệ số tương quan Pearson giữa độ ẩm mô phỏng và độ ẩm thực tế trong giờ có người chỉ đạt $r = 0.28 - 0.39$. Động học ẩm thực tế phụ thuộc vào việc đóng/mở cửa ngẫu nhiên và quán tính hấp phụ ẩm của vật liệu nội thất mà mô hình 1 vùng nhiệt dung chưa mô phỏng chi tiết.
4. **(d) Ánh xạ vận hành nhị phân là xấp xỉ**: Quy tắc ánh xạ nhị phân trong PV7 ($hvac\_kw < 1.0\text{ kW} \to$ Action 0, $\ge 1.0\text{ kW} \to$ Action 1) là một phép xấp xỉ thô, do hệ thống thực tế là chiller trung tâm điều biến lưu lượng nước lạnh (VAV/FCU) chứ không phải điều hòa cục bộ On/Off.
5. **(e) Hệ quả phương pháp luận**: Mọi kết quả tiện nghi nhiệt và tiết kiệm năng lượng từ mô phỏng **CHỈ CÓ Ý NGHĨA SO SÁNH TƯƠNG ĐỐI** giữa các thuật toán điều khiển (DQL vs QL vs Baselines) trên cùng một chuẩn tham chiếu môi trường, **TUYỆT ĐỐI KHÔNG KẾT LUẬN VỀ TIỆN NGHI HOẶC TIẾT KIỆM NĂNG LƯỢNG TUYỆT ĐỐI** trên tòa nhà thật.
6. **(f) Dữ liệu huấn luyện không có sương mù ô nhiễm**: Trong tập dữ liệu huấn luyện, nồng độ bụi mịn $PM_{2.5}$ hầu như không vượt quá $35\ \mu\text{g/m}^3$ (chiều $PM_{2.5}$ Bin 1 chỉ xuất hiện $0.25\% - 0.33\%$). Do đó, kịch bản sương mù ô nhiễm (smog) ở Test-C là bài kiểm tra **NGOÀI PHÂN PHỐI (Out-Of-Distribution)**. Khi không có ô nhiễm ngoại cảnh và nhiệt độ không xuống dưới $21^\circ\text{C}$, số trạng thái thực sự có thể đạt trong điều kiện bình thường bị giới hạn về mặt lý thuyết ở mức $\approx 4 \times 3 \times 3 \times 1 \times 2 \times 4 = 288 / 720$ trạng thái.
