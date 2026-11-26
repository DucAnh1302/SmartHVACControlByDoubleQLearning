# CƠ SỞ KHOA HỌC VÀ THIẾT KẾ HỆ THỐNG (SYSTEM DESIGN RATIONALE)

> **Mục tiêu**: Thuyết minh cơ sở lý luận, phương pháp luận toán học và nguyên lý vật lý cho các quyết định thiết kế không gian trạng thái, phân ranh ngưỡng tải nhiệt nội sinh và định cỡ công suất trong đề tài "Smart HVAC Control by Double Q-Learning".
>
> **Nguyên tắc Liêm chính**: Mọi quyết định thiết kế phải dựa trên đặc tính vật lý của công trình và phân phối thống kê của dữ liệu thực nghiệm, hoàn toàn độc lập với việc thuật toán điều khiển nào đạt điểm cao hơn.

---

## 1. Cơ sở lựa chọn ngưỡng phân ranh tải nhiệt nội sinh (Load Bin Threshold: 0.5 kW)

Trong không gian 720 trạng thái rời rạc của bài toán điều khiển HVAC:
- Chiều thứ 5 là **Tải nhiệt nội sinh (Internal Heat Load)** với 2 bin: `Low Load` (Bin 0) và `High Load` (Bin 1).
- Trong các tài liệu và mã nguồn ban đầu của repo gốc, ngưỡng này được đặt là **`3.0 kW`**.

### 1.1. Phân tích nguyên nhân không phù hợp của ngưỡng cũ (3.0 kW):
- Ngưỡng $3.0\text{ kW}$ trong repo gốc được lấy cảm quan từ dữ liệu thô của cả phân khu mở tầng 2 (`2018_F2_Z1`), nơi có tổng công suất thiết bị lên tới $13.36\text{ kW}$ (phục vụ hàng chục nhân viên).
- Khi bài toán mô phỏng được chuẩn hóa về quy mô phòng văn phòng tiêu chuẩn $50.0\text{ m}^2$, tổng tải nội sinh thiết kế cực đại theo tiêu chuẩn ASHRAE 90.1 ($35\text{ W/m}^2$) chỉ là:
  $$P_{target,max} = 0.035\text{ kW/m}^2 \times 50.0\text{ m}^2 = 1.750\text{ kW}$$
- Nếu tiếp tục giữ ngưỡng cũ $3.0\text{ kW}$ trên phòng $50\text{ m}^2$, phụ tải phòng thực tế ($0.03 - 1.75\text{ kW}$) sẽ **100% rơi vào Bin 0** (`< 3.0 kW`), khiến Bin 1 (`>= 3.0 kW`) trở thành một **chiều chết (dead dimension)**, làm suy biến một nửa không gian trạng thái (chỉ còn $720 / 2 = 360$ trạng thái thực tế có thể xuất hiện).

### 1.2. Phân tích phân phối thực nghiệm trên 60% Train Set (1,060 giờ):
Khảo sát phân phối phụ tải nội sinh đã chuẩn hóa quy mô phòng trên toàn bộ 1,060 giờ của tập huấn luyện (60% đầu `2018_F2_Z1`) cho thấy một phân phối **hai đỉnh rõ rệt (bimodal distribution)**:

| Trạng thái phòng | Số giờ | Tỷ lệ (%) | Phụ tải Trung bình (kW) | Phụ tải Trung vị (kW) | Phân vị 5% - 95% (kW) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Không có người (Unoccupied / Night)** | 643 | 60.66% | **0.059** | **0.026** | 0.023 – 0.038 |
| **Có người làm việc (Occupied / Day)** | 417 | 39.34% | **1.440** | **1.668** | 0.852 – 1.750 |
| **Toàn bộ tập huấn luyện** | 1,060 | 100.00% | 0.603 | 0.033 | 0.023 – 1.750 |

### 1.3. Ma trận nhầm lẫn & Mức độ trùng khớp với chỉ báo có người:
Để kiểm tra tính đại diện khách quan của ngưỡng $0.5\text{ kW}$, ta lập bảng đối chiếu chéo (contingency matrix) giữa sự kiện $Load \ge 0.5\text{ kW}$ và sự kiện phòng có người ($Occupancy > 0$) trên toàn bộ 1,060 giờ của tập Train:

| Thực tế phòng \ Ngưỡng tải | $Load < 0.5\text{ kW}$ (Bin 0) | $Load \ge 0.5\text{ kW}$ (Bin 1) | Tổng số giờ |
|---|:---:|:---:|:---:|
| **Phòng trống ($Occupancy = 0$)** | **TN = 624 (58.87%)** | FP = 19 (1.79%) | 643 (60.66%) |
| **Có người ($Occupancy > 0$)** | FN = 15 (1.42%) | **TP = 402 (37.92%)** | 417 (39.34%) |
| **Tổng số giờ** | 639 (60.28%) | 421 (39.72%) | **1,060 (100.0%)** |

- **Độ trùng khớp tổng thể (Overall Agreement)**:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Tổng}} = \frac{402 + 624}{1,060} = \mathbf{96.79\%}$$
- **Ý nghĩa kỹ thuật**:
  - Ngưỡng $0.5\text{ kW}$ đóng vai trò như một **chỉ báo vật lý gián tiếp về sự hiện diện của con người** (trùng khớp tới $96.8\%$). Khi có người vào phòng bật máy tính và chiếu sáng, phụ tải điện lập tức vượt ngưỡng $0.5\text{ kW}$.
  - Sai số phân loại rất thấp: Chỉ $1.79\%$ số giờ phòng trống có thiết bị chạy ngầm để quên vượt $0.5\text{ kW}$ (FP) và $1.42\%$ số giờ người vừa vào phòng chưa bật thiết bị (FN).
  - Quyết định đặt ngưỡng $0.5\text{ kW}$ là một **ranh giới Bayes tối ưu** phân tách chế độ chờ và chế độ hoạt động, hoàn toàn khách quan và độc lập với thuật toán điều khiển RL.

---

## 2. Loại bỏ các nhãn mô tả cũ sai lệch về số người

- Trong một số tài liệu cũ, chiều tải thường bị gán nhãn không chính xác kiểu `"<5 người"` hoặc `">10 người"`.
- Thực tế trong kiến trúc Digital Twin hiện đại:
  - Tải nhiệt nội sinh phản ánh **công suất điện của ổ cắm và đèn chiếu sáng** (`plug_kw + light_kw`), được đo lường bằng đơn vị $\text{kW}$.
  - Chiều tải này **mang chủ yếu thông tin có người/không có người (trùng 96.8%)**, giúp bộ điều khiển RL nhận biết trạng thái phòng để đưa ra hành động điều hòa phù hợp.
  - Số lượng người trong phòng là một biến vật lý độc lập (`occupancy_count`), được đưa trực tiếp vào phương trình cân bằng nhiệt qua nhiệt hiện sinh học ($0.075\text{ kW/người}$) và cân bằng $\text{CO}_2$ ($18,000\text{ ppm}\cdot\text{m}^3/\text{h/người}$).
  - Do đó, mọi mô tả không gian trạng thái trong Word, PowerPoint, file Excel và README bắt buộc phải dùng nhãn chuẩn: **`Tải nội sinh thấp (< 0.5 kW)`** và **`Tải nội sinh cao (>= 0.5 kW)`** (với chú thích: mang chủ yếu thông tin có người/không có người, trùng khớp 96.8%).

---

## 3. Khảo sát Độ phủ Không gian 720 Trạng thái trên Toàn bộ Ngày Train (State Space Coverage Diagnostic)

Chạy thử nghiệm chính sách ngẫu nhiên đều (Uniform Random Action Rollout) trong **200 episode** (4,800 bước thời gian 1 giờ) lấy mẫu ngẫu nhiên trên **TẤT CẢ** 44 ngày của tập huấn luyện (bao gồm 31 ngày thường và 13 ngày cuối tuần):
- **Tỷ lệ episode rơi vào cuối tuần**: 55 / 200 episode ($27.5\%$), phản ánh sát thực tế phân bố lịch ngày ($13/44 = 29.5\%$).
- **Quy tắc cuối tuần**: Toàn bộ 24 giờ của các ngày thứ Bảy và Chủ Nhật đều được phân loại chính xác vào $\text{time\_bin} = 0$ (Off-Peak/Weekend tariff).
- **Số trạng thái duy nhất được ghé thăm**: **129 / 720 trạng thái** (đạt $17.92\%$).
- Toàn bộ dữ liệu chi tiết từng trạng thái được lưu trữ tại `result/final/diag_state_coverage.csv` và trực quan hóa tại `result/final/diag_state_coverage.png`.

### Phân bố tần suất theo từng chiều rời rạc (Ngưỡng cắt chuẩn hóa):

1. **Nhiệt độ (5 bins: 21, 24, 26, 28.5°C)**:
   - Bin 0 ($< 21.0^\circ\text{C}$): **0 bước (0.00%) — Dead Bin (Chiều chết vật lý)**.
   - Bin 1 ($[21.0, 24.0)^\circ\text{C}$): 264 bước ($5.50\%$).
   - Bin 2 ($[24.0, 26.0]^\circ\text{C}$): 2,149 bước ($44.77\%$).
   - Bin 3 ($(26.0, 28.5)^\circ\text{C}$): 1,906 bước ($39.71\%$).
   - Bin 4 ($\ge 28.5^\circ\text{C}$): 481 bước ($10.02\%$).
2. **Độ ẩm tương đối (3 bins: 40%, 65%)**:
   - Bin 0 ($< 40\%$): 148 bước ($3.08\%$).
   - Bin 1 ($[40, 65]\%$): 2,333 bước ($48.60\%$).
   - Bin 2 ($> 65\%$): 2,319 bước ($48.31\%$).
3. **Nồng độ CO2 (3 bins: 800, 1200 ppm)**:
   - Bin 0 ($< 800\text{ ppm}$): 3,472 bước ($72.33\%$).
   - Bin 1 ($[800, 1,200]\text{ ppm}$): 724 bước ($15.08\%$).
   - Bin 2 ($> 1,200\text{ ppm}$): 604 bước ($12.58\%$).
4. **Nồng độ bụi mịn PM2.5 (2 bins: 35 µg/m³)**:
   - Bin 0 ($\le 35\ \mu\text{g/m}^3$): 4,788 bước ($99.75\%$).
   - Bin 1 ($> 35\ \mu\text{g/m}^3$): 12 bước ($0.25\%$) — Rare Bin (Trạng thái hiếm).
5. **Tải nhiệt nội sinh (2 bins: 0.5 kW)**:
   - Bin 0 ($< 0.5\text{ kW}$): 2,947 bước ($61.40\%$) — Chế độ chờ đêm và cuối tuần.
   - Bin 1 ($\ge 0.5\text{ kW}$): 1,853 bước ($38.60\%$) — Chế độ hoạt động giờ làm việc.
6. **Thời gian / Giá điện (4 bins)**:
   - Bin 0 (Off-Peak / Đêm & Toàn bộ cuối tuần): 2,480 bước ($51.67\%$).
   - Bin 1 (Morning Ramp 06:00–09:00 ngày thường): 435 bước ($9.06\%$).
   - Bin 2 (Midday Peak 09:00–17:00 ngày thường): 1,160 bước ($24.17\%$).
   - Bin 3 (Evening Shoulder 17:00–22:00 ngày thường): 725 bước ($15.10\%$).

### Phân tích nguyên nhân vật lý của Dead Bin và Giới hạn Không gian Trạng thái:
1. **Tại sao Bin Nhiệt độ 0 (`< 21.0°C`) không bao giờ xuất hiện?**
   - Bangkok nằm trong đới khí hậu nhiệt đới xích đạo, nhiệt độ môi trường ngoài trời trung bình luôn dao động từ $26.0^\circ\text{C}$ đến $36.0^\circ\text{C}$.
   - Danh mục hành động của hệ thống điều hòa HVAC được thiết kế làm lạnh với nhiệt độ setpoint thấp nhất là $23.0^\circ\text{C}$ (chế độ MAX_COOL, Action 2). Hệ thống không có chế độ làm lạnh công nghiệp xuống dưới $21.0^\circ\text{C}$.
   - Khi không làm mát, nhiệt độ phòng tự nhiên luôn trôi về dải $27.0 - 36.5^\circ\text{C}$ (như đã chứng minh ở PV1).
   - Do đó, trong điều kiện vật lý thực tế của công trình tại Bangkok, nhiệt độ phòng **vật lý không thể tự hạ xuống dưới $21.0^\circ\text{C}$**.
2. **Giới hạn không gian trạng thái thực tế có thể đạt (Theoretical Reachability Bounding)**:
   - Do Bin Nhiệt độ 0 ($< 21^\circ\text{C}$) là chiều chết vật lý (chỉ còn 4 bins nhiệt độ hoạt động) và nồng độ $PM_{2.5}$ trong tập train không có sương mù ô nhiễm (gần như $100\%$ thuộc Bin 0), trong điều kiện vận hành thông thường, số trạng thái thực sự có thể đạt được bị giới hạn trần ở mức:
     $$N_{reachable,theoretical} \approx 4\ (\text{nhiệt}) \times 3\ (\text{ẩm}) \times 3\ (CO_2) \times 1\ (PM) \times 2\ (\text{tải}) \times 4\ (\text{giờ}) = \mathbf{288 / 720\text{ trạng thái}}$$
   - Mức độ bao phủ $129$ trạng thái đạt được trong 200 episode ngẫu nhiên tương ứng với $129 / 288 = \mathbf{44.79\%}$ tổng số trạng thái khả dĩ trong điều kiện tự nhiên không ô nhiễm. Kịch bản sương mù ô nhiễm ở Test-C sẽ kích hoạt các trạng thái thuộc Bin PM2.5 số 1 nằm ngoài phân phối này.
   - Khi không làm mát, nhiệt độ phòng tự nhiên luôn trôi về dải $27.0 - 36.5^\circ\text{C}$ (như đã chứng minh ở PV1).
   - Do đó, trong điều kiện vật lý thực tế của công trình tại Bangkok, nhiệt độ phòng **vật lý không thể tự hạ xuống dưới $21.0^\circ\text{C}$**. Đây là một phát hiện khách quan có giá trị khoa học cao.
2. **Tại sao Bin PM2.5 số 1 (`> 35 µg/m³`) là trạng thái hiếm ($0.33\%$)?**
   - Nồng độ bụi mịn môi trường Bangkok trong mùa mưa hè (tháng 7–8/2018) duy trì ở mức trung bình $25.0\ \mu\text{g/m}^3$.
   - Khi phòng kín có hệ thống lọc khí tuần hoàn (filter HEPA MERV-13 tích hợp trong các action template làm mát/lọc), nồng độ bụi trong phòng thường xuyên được giữ ở mức $10 - 25\ \mu\text{g/m}^3$. Chỉ khi mở cửa sổ trong thời điểm ngoại cảnh ô nhiễm đột xuất thì nồng độ mới vượt qua $35\ \mu\text{g/m}^3$.
