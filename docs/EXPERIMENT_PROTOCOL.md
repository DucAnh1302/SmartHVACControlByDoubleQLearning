# GIAO THỰC THỰC NGHIỆM ĐỐI CHIẾU (EXPERIMENT PROTOCOL - PROTOCOL LOCKED)

> **Mục tiêu**: Thiết lập và khóa cứng trước các định nghĩa, luật điều khiển baseline, không gian siêu tham số, cơ chế điều biến và nguyên tắc kiểm định thống kê TRƯỚC KHI chạy thực nghiệm chính thức (Phase 3).
> **Nguyên tắc Liêm chính**: Giao thức này được cam kết và gắn thẻ Git `protocol-locked` (bản gốc) và `protocol-locked-v2` (hiệu chỉnh vế 2 đã duyệt) trước khi kết quả kiểm thử Test-A/B/C được sinh ra. Tuyệt đối không thay đổi bất kỳ điều khoản nào dựa trên kết quả tập Test.

### Bản cập nhật Giao thức v2 (Đã phê duyệt trước khi chạy Test)
1. **Hiệu chỉnh Action 5 (`ACTIVE_VENT_PURIFY`)**: Kết hợp làm mát đồng thời tại ngưỡng $25.0^\circ\text{C}$ (`ac_mode="SET_25"`, `ac_setpoint_c=25.0`) song song với quạt gió tươi $500\text{ m}^3/\text{h}$ và lọc khí HEPA $450\text{ m}^3/\text{h}$, loại bỏ lỗi thổi khí nóng ngoại cảnh $32\text{--}36^\circ\text{C}$ làm tăng nhiệt phòng khi thông gió xả $CO_2$.
2. **Tín hiệu phản hồi Safety Gateway trong vòng lặp**: Truyền cờ `safety_violation=(safe_act != raw_action)` vào `reward_fn.calculate_reward(...)` ở cả huấn luyện và đánh giá để agent học tránh hành động nguy hiểm.
3. **Lịch suy giảm Epsilon hội tụ về $\epsilon_{\min}=0.02$**: Thay lịch cố định bị nghẽn bằng tỷ lệ mục tiêu $f_{\text{decay}} \in \{0.65, 0.75, 0.85\}$ sao cho $\epsilon_{\text{decay}} = (\epsilon_{\min}/\epsilon_0)^{1/(f_{\text{decay}} \cdot N_{\text{ep}})}$ trên ngân sách $N_{\text{ep}} \in \{1200, 2000, 3000\}$ và $\alpha \in \{0.05, 0.10, 0.15\}$.

---

## 1. Bốn Câu Hỏi Nghiên Cứu Cốt Lõi (Core Research Questions)

1. **RQ1 (Thiên lệch ước lượng)**: Thuật toán Double Q-Learning (DQL) có làm giảm thiên lệch ước lượng giá trị tối đại (Maximization Bias) so với Single Q-Learning (QL) trên bài toán điều khiển HVAC đa thiết bị hay không?
2. **RQ2 (Hiệu năng chính sách)**: Double Q-Learning có đem lại chính sách điều khiển tối ưu hơn Q-Learning trên 3 tiêu chí cốt lõi (Điện năng, Tiện nghi vi khí hậu/IAQ, Bảo vệ thiết bị) hay không?
3. **RQ3 (So sánh với Baseline)**: Các thuật toán Học tăng cường (DQL và QL) có vượt trội hơn các bộ điều khiển cơ sở kinh điển (Fixed Setpoint, Conventional Thermostat, Rule-Based Heuristic, Random) hay không?
4. **RQ4 (Độ bền vững & Chuyển dịch miền)**: Độ bền vững (Robustness) của chính sách học được khi môi trường gặp dịch chuyển miền (Domain Shift: tập năm 2019 Test-B; các kịch bản thời tiết cực đoan và ô nhiễm Test-C: Sóng nhiệt gay gắt, quá tải người, khói bụi mịn PM2.5)?

---

## 2. Ba Tiêu Chí Đánh Giá Cốt Lõi & Chỉ Số Đo Lường Vật Lý

> **Ràng buộc phương pháp luận quan trọng**: Các chỉ số về tiện nghi nhiệt và chất lượng không khí trong nhà (IAQ) chỉ được tính toán trong **giờ có người trong phòng** (`occupancy_count > 0`), vì việc điều hòa nhiệt độ và chất lượng không khí cho phòng trống không mang ý nghĩa phục vụ con người.

### 2.1. Tiêu chí 1: Tiết kiệm Điện năng Tiêu thụ
- **Điện năng tiêu thụ trung bình hàng ngày**: Đơn vị $\text{kWh/ngày}$ (tính tích phân công suất toàn bộ các thiết bị HVAC trong cả 24 giờ của mỗi ngày).

### 2.2. Tiêu chí 2: Tiện nghi Vi khí hậu & Chất lượng Không khí (IAQ)
*(Chỉ đo đạc trong các giờ có người hiện diện `occupancy_count > 0`)*:
- **Tỷ lệ vi phạm tiện nghi nhiệt**: $\%$ số giờ có người có nhiệt độ trong phòng nằm ngoài khoảng tiện nghi:
  $$\% \text{ Giờ vi phạm } T = \frac{\sum_{t \in \mathcal{T}_{occ}} \mathbb{I}(T_t < 24.0^\circ\text{C} \lor T_t > 26.0^\circ\text{C})}{|\mathcal{T}_{occ}|} \times 100\%$$
- **Tỷ lệ vi phạm nồng độ $CO_2$**: $\%$ số giờ có người có nồng độ $CO_2 > 1000.0\text{ ppm}$.
- **Tỷ lệ vi phạm bụi mịn $PM_{2.5}$**: $\%$ số giờ có người có nồng độ $PM_{2.5} > 35.0\text{ }\mu\text{g/m}^3$.
- **Tỷ lệ vi phạm độ ẩm tương đối**: $\%$ số giờ có người có $RH$ nằm ngoài khoảng $[40.0, 65.0]\%$.

### 2.3. Tiêu chí 3: Bảo vệ Tuổi thọ Thiết bị & Ổn định Hệ thống
- **Tần suất chuyển đổi hành động (Action Switching Frequency)**: Số lần thay đổi trạng thái điều khiển trung bình trong một ngày ($\text{switching/ngày}$). Giảm đóng ngắt máy nén và van cấp gió để bảo toàn cơ khí.
- **Tỷ lệ can thiệp của Safety Gateway (Safety Override Rate)**: $\%$ số bước mà hành động của Agent bị tầng Gateway an toàn chặn và cưỡng bức ghi đè để bảo vệ thiết bị hoặc con người.

### 2.4. Chỉ số Phụ trợ (Auxiliary Metric)
- **Điểm phần thưởng trung bình (Mean Episode Reward)**: Tổng điểm thưởng tích lũy trung bình mỗi episode. Cần lưu ý hàm phần thưởng là đại lượng do nhóm nghiên cứu thiết kế, không phải thước đo vật lý khách quan duy nhất để kết luận hiệu quả thực tế.

---

## 3. Danh Mục 6 Bộ Điều Khiển Tham Gia So Sánh (Bắt Buộc Đủ Cả 6)

Mọi bảng biểu và đồ thị so sánh chính trong báo cáo bắt buộc phải hiển thị đầy đủ 6 bộ điều khiển:
1. **Double Q-Learning (DQL)**: Thuật toán học bảng Q kép phân tách hành động chọn và hành động đánh giá ($720\text{ states} \times 12\text{ actions}$), tích hợp Safety Gateway trong vòng lặp.
2. **Single Q-Learning (QL)**: Thuật toán Q-learning kinh điển với bảng Q đơn, dùng chung hoàn toàn State, Action, Reward và môi trường mô phỏng.
3. **Fixed Setpoint (25°C Cool)**: Giữ cố định Action 1 (`COMFORT_COOL`: AC làm mát $25^\circ\text{C}$, quạt cấp gió thấp, lọc khí thấp). Đại diện cho thực tiễn vận hành văn phòng phổ biến hiện nay.
4. **Conventional Thermostat (Thermostat Khí hậu Nhiệt đới)**:
   - $T_{indoor} \ge 28.5^\circ\text{C} \rightarrow$ Action 2 (`MAX_COOL`).
   - $26.0^\circ\text{C} < T_{indoor} < 28.5^\circ\text{C} \rightarrow$ Action 1 (`COMFORT_COOL`).
   - $21.0^\circ\text{C} \le T_{indoor} \le 26.0^\circ\text{C} \rightarrow$ Action 0 (`ALL_OFF_STANDBY`, dải chết deadband, không sưởi).
   - $T_{indoor} < 21.0^\circ\text{C} \rightarrow$ Action 7 (`MAX_HEAT`, chỉ bật khi có đợt rét đột xuất).
5. **Rule-Based Smart HVAC (Bộ điều khiển Chuyên gia Heuristic)**:
   - Ưu tiên an toàn nhiệt: nếu $T \ge 28.5^\circ\text{C} \rightarrow$ Action 2; nếu $T < 21^\circ\text{C} \rightarrow$ Action 7.
   - Ưu tiên chất lượng không khí: nếu $CO_2 > 1200\text{ ppm} \rightarrow$ Action 5 (`ACTIVE_VENT_PURIFY`); nếu $PM_{2.5} > 35 \rightarrow$ Action 10 (`ECO_AIR_PURIFY`).
   - Ưu tiên độ ẩm: nếu $RH > 65\% \rightarrow$ Action 8 (`DEHUMIDIFY_ACTIVE`); nếu $RH < 40\% \rightarrow$ Action 9 (`HUMIDIFY_COMFORT`).
   - Giờ làm việc bình thường: nếu $T \in (26.0, 28.5] \rightarrow$ Action 1 (hoặc Action 3 trong giờ cao điểm điện); nếu $T \in [21, 24] \rightarrow$ Action 0.
   - Ban đêm: Action 11 (`NIGHT_SETBACK`).
6. **Random Policy**: Chọn ngẫu nhiên đều 1 trong 12 hành động phối hợp.

---

## 4. Không Gian Siêu Tham Số, Ngân Sách & Kế Hoạch Huấn Luyện

### 4.1. Không gian tìm kiếm lưới (Hyperparameter Grid - Protocol v2)
Áp dụng lưới tìm kiếm giống nhau hoàn toàn cho cả DQL và QL trên ngân sách bình đẳng:
- Tốc độ học $\alpha \in \{0.05, 0.10, 0.15\}$ (3 giá trị; chọn trên Validation để tránh dao động phân kỳ ở $\alpha=0.20$ tại $3,000$ episodes).
- Tỷ lệ phân rã khám phá $f_{\text{decay}} \in \{0.65, 0.75, 0.85\}$ (3 giá trị), quy đổi tường minh thành $\epsilon_{\text{decay}} = (0.02 / 1.0)^{1 / (f_{\text{decay}} \cdot N_{\text{ep}})}$ đảm bảo $\epsilon$ đạt $\epsilon_{\min} = 0.02$ trước đuôi khai thác tham lam ($15\text{--}35\%$ cuối).
- Số lượng episode huấn luyện $N_{\text{ep}} \in \{1200, 2000, 3000\}$ (3 giá trị).
- Hệ số chiết khấu $\gamma = 0.95$ (cố định theo đặc tả).
- Giá trị $\epsilon_{\min} = 0.02$, $\epsilon_{\text{start}} = 1.0$.

Tổng cộng có $3 \times 3 \times 3 = 27$ cấu hình siêu tham số ($162$ lần chạy grid trên Validation).

### 4.2. Quy trình lựa chọn trên tập Validation (Validation Grid Tuning)
- Mỗi cấu hình được chạy với 3 seed ngẫu nhiên độc lập (`seed=101, 102, 103`), mỗi seed đánh giá 8 episodes trên tập **Validation** ($27 \times 3 = 81$ lần chạy cho mỗi thuật toán, tổng cộng 162 lần chạy).
- Tiêu chí lựa chọn: Cấu hình có phần thưởng trung bình cao nhất trên tập Validation được chọn làm siêu tham số tối ưu độc lập cho DQL và QL.

### 4.3. Ngân sách Thời gian & Thực thi
- Thời gian đo đạc thực tế (sau tối ưu vectorized cache): trung bình $\sim 0.018\text{ s/episode}$ ($\sim 37\text{ s}$ cho $2,000$ episodes trên 1 luồng CPU).
- Lưới 162 lần chạy chạy song song qua `ProcessPoolExecutor(max_workers=8)` tiêu tốn $\sim 12\text{--}15\text{ phút}$; huấn luyện 10 seeds cuối (20 mô hình) chạy song song tiêu tốn $\sim 2\text{--}3\text{ phút}$.
- Tổng thời gian toàn bộ thực nghiệm $\sim 15\text{--}20\text{ phút}$, nằm hoàn toàn trong ngưỡng an toàn ($< 3\text{ giờ}$).

### 4.4. Đánh giá Kiểm thử Cuối cùng (10 Seeds, Run-Once Rule)
- 10 seed ngẫu nhiên độc lập (`seed = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9`) được dùng để huấn luyện mô hình chính thức trên tập **Train**.
- Đánh giá duy nhất một lần trên:
  1. **Test-A (2018_F2_Z1, 20% cuối)**: Tổng quát hóa cùng miền (In-distribution).
  2. **Test-B (2019_F2_Z1)**: Dịch chuyển phân phối xuyên năm (Cross-year domain transfer).
  3. **Test-C (Môi trường cực đoan)**: 3 kịch bản stress testing (`heatwave`, `large_occupancy`, `smog_pollution`).

---

## 5. Phương Pháp Kiểm Định Thống Kê & Phân Tích Phương Sai

So sánh giữa các bộ điều khiển được thực hiện ghép cặp theo từng seed (Paired evaluation):
1. **So sánh Cặp DQL vs QL (So sánh Cốt lõi)**:
   - **Paired Student's t-test**: Kiểm tra ý nghĩa thống kê về độ chênh lệch kỳ vọng điểm thưởng và điện năng giữa DQL và QL.
   - **Wilcoxon Signed-Rank Test**: Kiểm định phi tham số cho dữ liệu ghép cặp không giả định phân phối chuẩn.
   - **Cohen's d ($d_z$ ghép cặp)**: Đo lường quy mô ảnh hưởng chuẩn hóa trên hiệu ghép cặp $\Delta_s = x_{\text{DQL}, s} - x_{\text{QL}, s}$:
     $$d_z = \frac{\bar{\Delta}}{s_{\Delta}}$$
     Trong đó $|d_z| < 0.2$ (không đáng kể), $0.2 \le |d_z| < 0.5$ (nhỏ), $0.5 \le |d_z| < 0.8$ (trung bình), $|d_z| \ge 0.8$ (lớn).
   - **Bootstrap 95% Confidence Interval**: Rút mẫu có hoàn lại 10,000 lần để ước lượng khoảng tin cậy của chênh lệch hiệu năng trung bình $(\mu_{\text{DQL}} - \mu_{\text{QL}})$.
2. **So sánh với các Baseline (Multi-Comparison Correction)**:
   - Sử dụng hiệu chỉnh **Holm-Bonferroni** để kiểm soát tỷ lệ sai số gia đình (FWER) khi kiểm định đồng thời với các baseline.
3. **Phân tích Phương sai Giữa các Seed (Variance Analysis)**:
   - Báo cáo phương sai $\sigma^2$ và độ lệch chuẩn $s$ giữa 10 seeds của DQL so với QL.
   - Kiểm định **Levene's test** để xác định xem DQL có làm giảm độ phân tán/biến thiên chính sách giữa các lần khởi tạo seed so với QL hay không.

---

## 6. Quy Tắc Diễn Giải Kết Quả Đã Khóa (Pre-Registered Interpretation Branches)

Kết luận trong báo cáo và slide thuyết trình bắt buộc phải tuân theo 1 trong 3 nhánh định trước dựa trên dữ liệu thực tế, tuyệt đối không được bẻ cong:

- **Nhánh 1 (Branch 1 - DQL thắng có ý nghĩa thống kê)**:
  - *Điều kiện*: $p < 0.05$ (sau hiệu chỉnh) và kích thước ảnh hưởng Cohen's $d \ge 0.5$ nghiêng về DQL.
  - *Kết luận*: "Double Q-Learning cải thiện có ý nghĩa thống kê về hiệu năng điều khiển / điện năng trong điều kiện mô phỏng này; cơ chế tách rời chọn và đánh giá hành động phát huy tác dụng rõ rệt."
- **Nhánh 2 (Branch 2 - DQL và QL tương đương)**:
  - *Điều kiện*: $p \ge 0.05$ hoặc $|d| < 0.2$.
  - *Kết luận*: "Double Q-Learning và Single Q-Learning đạt hiệu năng tương đương, không có sự khác biệt có ý nghĩa thống kê; lợi thế lý thuyết của Double Q-Learning về việc giảm thiên lệch ước lượng chưa chuyển hóa thành ưu thế vượt trội ở cấp độ chính sách trên không gian trạng thái rời rạc này."
- **Nhánh 3 (Branch 3 - QL thắng có ý nghĩa thống kê)**:
  - *Điều kiện*: $p < 0.05$ và $d \le -0.5$ nghiêng về QL.
  - *Kết luận*: "Single Q-Learning đạt hiệu năng cao hơn Double Q-Learning trong bài toán này. Nguyên nhân khả dĩ từ dữ liệu: tốc độ cập nhật của bảng Q kép chậm hơn trong ngân sách episode hạn chế hoặc cấu trúc phần thưởng phạt quá mạnh khiến việc ước lượng dưới của DQL dẫn đến chính sách quá thận trọng."
- **So sánh với Baseline (Liêm chính Khoa học)**:
  - Nếu Rule-Based hoặc Fixed Setpoint đạt điểm tốt hơn hoặc tiết kiệm điện hơn cả hai thuật toán RL, nhóm phải **nêu thẳng sự thật**, không che giấu, và phân tích nguyên nhân kỹ thuật (không gian 720 trạng thái còn thô, độ phủ trạng thái trong quá trình học khoảng $18\text{--}45\%$, sự can thiệp của Safety Gateway giới hạn không gian hành động tự do).

---

## 7. Thí Nghiệm Kiểm Chứng Thiên Lệch Ước Lượng (Maximization Bias Verification)

Để trả lời trọn vẹn câu hỏi RQ1, hai bài thực nghiệm kiểm chứng thiên lệch được triển khai:
1. **Bài toán Đồ chơi có Lời giải Giải tích (Toy MDP - Sutton & Barto Ví dụ 6.7)**:
   - Mô phỏng MDP 2 trạng thái: Trạng thái A có 2 hành động: "trái" (sang B, reward 0) và "phải" (kết thúc, reward 0).
   - Tại trạng thái B có 10 hành động, mỗi hành động sinh phần thưởng ngẫu nhiên $R \sim \mathcal{N}(-0.1, 1.0)$.
   - Giá trị kỳ vọng thật của mọi hành động tại B là $-0.1 < 0$. Do đó hành động tối ưu tại A là "phải" (giá trị 0).
   - Tuy nhiên, toán tử cực đại $\max_a Q(B, a)$ của Q-Learning bị thiên lệch dương ($\mathbb{E}[\max X_i] > 0$), dẫn đến việc Agent chọn hành động "trái" với xác suất rất cao (>50% ở giai đoạn đầu).
   - Double Q-Learning sử dụng bảng Q kép độc lập sẽ loại bỏ thiên lệch này và duy trì xác suất chọn "trái" tiệm cận mức khám phá ngẫu nhiên ($\epsilon / 2 = 5\%$).
   - Kết quả được vẽ thành biểu đồ tỷ lệ chọn hành động "trái" theo episode trung bình qua 2,000 lần chạy độc lập (300 episodes/lần).
2. **Đo đạc Thiên lệch trên Môi trường HVAC**:
   - Định nghĩa độ thiên lệch:
     $$\text{Bias}(s_0) = \max_a Q(s_0, a) - G_{MC}(s_0)$$
   - Trong đó $G_{MC}(s_0)$ là giá trị hoàn trả chiết khấu trung bình thu được từ 50 lượt rollout Monte Carlo trên 20 trạng thái lấy mẫu từ tập Test-A của chính sách tham lam (greedy policy) xuất phát từ cùng trạng thái ban đầu $s_0$.
   - Giá trị $Q(s_0, a)$ của DQL là giá trị trung bình kết hợp $\frac{Q_A(s_0, a) + Q_B(s_0, a)}{2}$.
   - Độ thiên lệch dương $(\text{Bias} > 0)$ phản ánh hiện tượng thổi phồng giá trị (overestimation).
   - Báo cáo giá trị trung bình $\text{Bias}_{\text{QL}}$ và $\text{Bias}_{\text{DQL}}$ kèm khoảng tin cậy qua các seeds.

---

## 8. Khóa Cứng Cấu Hình Vật Lý & Mã Nguồn

Các thông số sau đây tiếp tục được duy trì khóa cứng:
- Thể tích phòng: $140.0\text{ m}^3$ ($50\text{ m}^2 \times 2.8\text{ m}$).
- Công suất AC: HIGH $1.59\text{ kW}$ (4.93 kW nhiệt), MED $1.01\text{ kW}$, LOW $0.52\text{ kW}$, COP 3.2. Sizing ratio = $1.3004$.
- Tham số cuộn lạnh: $T_{coil} = 12.0^\circ\text{C}$, $V_{coil} = 0.08\text{ m}^3\text{/s/kW}$, nền ẩm ngoại cảnh $68.0\%$ (`SIMULATED_ASSUMED`).
- Ranh giới trạng thái: $0.5\text{ kW}$ cho tải nội sinh.
