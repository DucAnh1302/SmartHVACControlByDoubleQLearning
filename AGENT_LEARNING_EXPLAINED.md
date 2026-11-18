# AGENT LEARNING EXPLAINED — DOUBLE Q-LEARNING FOR MULTI-DEVICE HVAC CONTROL

> [!IMPORTANT]
> **Tuyên bố Phạm vi Nghiên cứu (`RESEARCH_SIMULATION_ONLY`):**  
> Tài liệu này mô tả thuật toán Tabular Double Q-Learning trong môi trường mô phỏng Digital Twin của dự án. Thuật toán được kiểm nghiệm trên các phương trình truyền nhiệt, ẩm học tâm trắc và chất lượng không khí toán học, **tuyệt đối không phải hệ thống điều khiển đã triển khai trên tòa nhà thực tế**.

---

## 1. Cấu trúc Một Tập Mô Phỏng (Episode & Step Chronology)

Mỗi episode mô phỏng trong hệ thống đại diện cho một chu kỳ vận hành liên tục:
- **Độ dài tập (Episode Length):** $48\text{ bước}$ (tương đương 48 giờ liên tục, bao gồm cả ngày làm việc và cuối tuần hoặc chu kỳ ngày - đêm).
- **Thời lượng mỗi bước ($\Delta t$):** $1.0\text{ giờ}$ mô phỏng vật lý (trong GUI, tương ứng với 100ms - 2000ms wall-clock time tùy cấu hình tốc độ).
- **Quy trình thực thi một bước ($\tau \to \tau + 1$):**
  1. **Quan sát Trạng thái Hiện tại ($s_\tau$):** Vector 6 chiều liên tục được lượng tử hóa thành chỉ số trạng thái $s_\tau \in \{0, 1, \dots, 719\}$.
  2. **Lựa chọn Hành động Đề xuất ($a_{candidate}$):** Agent sử dụng chính sách $\epsilon$-greedy dựa trên bảng giá trị kết hợp $Q_{comb}(s, a) = \frac{Q_1(s, a) + Q_2(s, a)}{2}$.
  3. **Kiểm tra và Can thiệp An toàn (Safety Gateway):** $a_{candidate}$ được chuyển qua `MultiDeviceSafetyGateway`. Nếu vi phạm an toàn nhiệt, khí ngạt, nguy cơ thời tiết hoặc chu kỳ máy nén, Gateway sẽ override thành $a_{safe}$.
  4. **Bước Chuyển tiếp Môi trường (Environment Step):** Hệ phương trình vi phân vật lý (nhiệt, ẩm, CO₂, PM2.5, công suất) giải tích phân số trong $\Delta t = 1.0\text{h}$, sinh ra trạng thái tiếp theo $s_{\tau+1}$.
  5. **Tính toán Phần thưởng Phân rã (Reward Calculation):** Hàm thưởng phân rã đa mục tiêu tính toán $r_\tau = R_{temp} + R_{humidity} + R_{co2} + R_{air\_quality} + R_{energy} + R_{switching} + R_{safety}$.
  6. **Cập nhật Bảng Giá trị (Q-Update):** Nếu ở `Online Learning Mode`, chuyển tiếp $(s_\tau, a_{safe}, r_\tau, s_{\tau+1})$ được cập nhật vào $Q_1$ hoặc $Q_2$. Nếu ở `Evaluation Mode`, hai bảng $Q$ hoàn toàn được giữ đóng băng.

---

## 2. Bản chất Toán học của Double Q-Learning

### 2.1. Vấn đề Độ lệch Tối đa hóa (Maximization Bias) trong Q-Learning Truyền thống
Trong Q-Learning tiêu chuẩn (Single Q-Learning), toán tử cập nhật sử dụng giá trị cực đại của bảng Q để vừa chọn hành động vừa ước lượng giá trị mục tiêu:

$$Y_t^{Q} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$$

Do có sai số ngẫu nhiên trong ước lượng $\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$, toán tử $\max$ luôn tạo ra độ lệch dương có hệ thống, khiến Agent đánh giá quá cao các hành động rủi ro (overestimation).

### 2.2. Cơ chế Hai Bảng Độc lập trong Double Q-Learning
Double Q-Learning giải quyết triệt để vấn đề này bằng cách phân tách việc **chọn hành động tốt nhất** và **đánh giá giá trị của hành động đó** thành 2 bảng giá trị độc lập $Q_1$ và $Q_2$:

Tại mỗi bước học, một biến ngẫu nhiên đồng đều $p \sim \mathcal{U}(0, 1)$ quyết định bảng nào được cập nhật:

- **Nếu $p < 0.5$ (Cập nhật $Q_1$):**
  $$a^* = \operatorname{argmax}_{a} Q_1(S_{t+1}, a)$$
  $$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_2(S_{t+1}, a^*) - Q_1(S_t, A_t) \right]$$

- **Nếu $p \ge 0.5$ (Cập nhật $Q_2$):**
  $$b^* = \operatorname{argmax}_{a} Q_2(S_{t+1}, a)$$
  $$Q_2(S_t, A_t) \leftarrow Q_2(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_1(S_{t+1}, b^*) - Q_2(S_t, A_t) \right]$$

Bảng $Q_1$ tìm hành động có giá trị kỳ vọng lớn nhất, nhưng bảng $Q_2$ mới là bảng lượng giá độc lập cho hành động đó. Điều này triệt tiêu hoàn toàn Maximization Bias.

---

## 3. Khám phá (Exploration) vs Khai thác (Exploitation)

- **Chiến lược $\epsilon$-greedy:**
  - Với xác suất $\epsilon$: Agent chọn ngẫu nhiên một hành động $a \in \{0, 1, \dots, 11\}$ để khám phá không gian phối hợp thiết bị mới.
  - Với xác suất $1 - \epsilon$: Agent khai thác tri thức hiện tại bằng cách chọn $a^* = \operatorname{argmax}_a \frac{Q_1(s, a) + Q_2(s, a)}{2}$.
  - Giải quyết hòa điểm (Tie-breaking): Nếu nhiều hành động có cùng giá trị $Q$ cực đại, Agent bốc thăm ngẫu nhiên có định hướng hạt giống (`seed`) thay vì chọn mặc định hành động đầu tiên, ngăn chặn thiên kiến cố định.
- **Suy giảm Epsilon ($\epsilon$-decay):**
  $$\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \epsilon_{decay})$$
  Với $\epsilon_0 = 1.0$, $\epsilon_{decay} = 0.992$, $\epsilon_{min} = 0.02$. Trong 400 episodes, Agent dịch chuyển mượt mà từ khám phá toàn diện sang khai thác chuyên sâu.
- **Không lấy mẫu RNG hai lần:** Trạng thái khám phá được lưu trực tiếp vào cờ `agent.last_action_is_exploring` tại đúng thời điểm gọi hàm `select_action()`, đảm bảo nhãn hiển thị trên Decision Trace phản ánh trung thực 100% quyết định nội tại của Agent.

---

## 4. Chế độ Vận hành: Evaluation vs Online Learning

| Tiêu chí | Chế độ Đánh giá (Evaluation Mode) | Chế độ Học trực tuyến (Online Learning Mode) |
| :--- | :--- | :--- |
| **Mục đích** | Đánh giá chính sách cố định, biểu diễn trực quan | Tiếp tục thích nghi chính sách khi môi trường đổi |
| **Giá trị Epsilon** | Cưỡng bức $\epsilon = 0$ (Khai thác tham lam hoàn toàn) | Sử dụng $\epsilon$ hiện tại của Agent |
| **Cập nhật Q-Table** | **KHÔNG** (Q1, Q2 giữ nguyên 100%, bộ nhớ đóng băng) | **CÓ** ($Q_1$ hoặc $Q_2$ được cập nhật tại mỗi bước) |
| **Bộ đếm cập nhật** | Giữ nguyên `#0` | Tăng dần `+1` sau mỗi transition thành công |
| **Hỗ trợ Bộ điều khiển** | Tất cả 6 bộ điều khiển (DQL, SQL, Heuristic, Baseline)| **Chỉ Double Q-Learning và Single Q-Learning** |

---

## 5. Lưu trữ và Kiểm soát Phiên bản Mô hình (Model Checkpoint Metadata)

Mỗi mô hình huấn luyện được lưu dưới dạng file nhị phân pickle kèm từ điển metadata định danh:
- `schema_version`: `2.0-multi-device-720s12a`
- `algorithm`: `DoubleQLearning`
- `n_states`: `720`
- `n_actions`: `12`
- `seed`: `42`
- `training_episodes`: `400`
- `created_at`: Dấu thời gian ISO 8601 chuẩn UTC.

Khi tải mô hình, hàm `load_checkpoint` kiểm tra nghiêm ngặt `schema_version`. Nếu phát hiện phiên bản schema không tương thích (ví dụ checkpoint cũ 600 trạng thái x 5 hành động), hệ thống sẽ ném ngoại lệ tường minh `ValueError`, ngăn chặn việc nạp sai lệch trọng số vào Q-table mới.
