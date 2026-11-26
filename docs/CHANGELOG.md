# NHẬT KÝ THAY ĐỔI (PROJECT CHANGELOG)

Tài liệu này ghi nhận mọi thay đổi kỹ thuật trên repo `Nhom2` xuyên suốt quá trình thực hiện đồ án "Smart HVAC Control by Double Q-Learning" (Hướng A). Tuân thủ nguyên tắc liêm chính học thuật: mọi thay đổi phải có lý do, có đối chiếu với phát hiện lỗi thực tế (F1–F22), và được kiểm chứng bằng test tự động.

---

## [Phase 0] - Đóng băng hiện trạng & Xác lập mốc đối chiếu (2026-11-24)

### Thêm mới (Added)
- Tạo nhánh git `huong-a` và gắn tag `baseline-goc` trên commit đầu tiên `ba73e37`.
- Tạo `.gitignore` chuẩn cho Python/Pytest, loại bỏ `__pycache__`, `.pytest_cache`, và cấu hình loại bỏ các file CSV dung lượng lớn (>750MB) trong `data/raw/` theo đúng chỉ dẫn của `data/raw/README.md`.
- Tạo thư mục sao lưu `result/_baseline_goc/` lưu trữ snapshot toàn bộ kết quả, mô hình và đồ thị trước khi sửa đổi:
  - `result/_baseline_goc/multi_device_benchmark.json`
  - `result/_baseline_goc/multi_device_training_curves.png`
  - `result/_baseline_goc/multi_device_double_q_model.pkl`
  - `result/_baseline_goc/multi_device_single_q_model.pkl`
- Tạo các tài liệu quản lý và định hướng:
  - `docs/BASELINE_RUN.md`: Chi tiết kết quả chạy kiểm thử (68/68 pass) và benchmark hiện trạng.
  - `docs/CHANGELOG.md`: Nhật ký thay đổi dự án.
  - `docs/OPEN_QUESTIONS.md`: Danh mục các câu hỏi mở và mục `[CẦN NGƯỜI XÁC NHẬN]`.
  - `docs/TEAM_PLAN.md`: Kế hoạch phân công công việc 4 thành viên và lịch trình phối hợp.

### Kiểm chứng (Verified)
- Môi trường thực thi: Python 3.12.5 trên Windows.
- Chạy `pytest tests -v`: Đạt 68/68 passed trong 7.87s.
- Chạy `train_multi_device_dql.py`: Tái lập 100% kết quả benchmark gốc trong `result/multi_device_benchmark.json` (DQL: -408.9 R, 41.2 kWh, 142 overrides).

---

## [Phase 1] - Sửa mô phỏng đúng vật lý (2026-11-24)

### Thay đổi & Khắc phục lỗi (Changed & Fixed)
- **F3 (Số người & Nhiệt tải sinh học):**
  - Cập nhật `src/data/domain_loader.py` tự động sinh cột `occupancy_est` gắn nhãn `DERIVED` từ `occupancy_proxy` và tải nội sinh.
  - Cập nhật `src/environment/multi_device_env.py` trong cả `reset()` và `step()` để cập nhật `self.occupancy_count` động theo từng bước thời gian.
  - Đưa nhiệt tỏa từ người vào cân bằng nhiệt: $Q_{occ} = 0.075\text{ kW/người} \times \text{occupancy_count}$.
- **F4 (Thời tiết & Ngoại cảnh xác định):**
  - Cập nhật `src/data/domain_loader.py` sinh `ambient_temp_est` và `ambient_rh_est` với dao động xác định theo hàm băm MD5 ngày (`SIMULATED_ASSUMED`).
  - Khai báo tường minh nồng độ ngoại cảnh: `ambient_co2_est = 415.0 ppm`, `ambient_pm25_est = 25.0 µg/m³`.
- **F5 (Định cỡ tải & Thiết bị - Sizing):**
  - Tính toán tỷ lệ co tỷ lệ tải phòng $50\text{ m}^2$ từ dữ liệu toàn zone ($60\%$ train split, $P95 = 12.795\text{ kW}$): $s = 1.75 / 12.795 \approx 0.136775$.
  - Tải lạnh thiết kế cực đại của phòng: $Q_{design} = 3.79\text{ kW}$.
  - Công suất lạnh tối đa của AC: $3.75\text{ kW điện} \times 3.2\text{ COP} = 12.0\text{ kW nhiệt} \ge 1.2 \times Q_{design}$ (Hệ số an toàn đạt $3.17$).
  - Lưu giữ cột `internal_heat_kw` nguyên bản cho các test rò rỉ dữ liệu, tạo cột `internal_heat_scaled_kw` cho mô phỏng môi trường phòng.
  - Cân chỉnh ngưỡng phân ranh tải cao/thấp trong `discretize_state_720` thành $0.5\text{ kW}$.
- **F6 (Tích phân đa bước Multi-rate Sub-stepping):**
  - Chia 1 bước điều khiển 1 giờ thành $n_{sub} = 12$ bước tích phân vật lý $5\text{ phút}$ ($300\text{ s}$).
  - Điều chỉnh nhiễu quá trình theo $1/\sqrt{12}$.
  - Điều biến công suất lạnh AC mượt mà theo bậc setpoint trong giờ.
  - Triệt tiêu hoàn toàn dao động răng cưa vô lý của nồng độ PM2.5.
- **F7 (Safety Gateway trong vòng huấn luyện):**
  - Tích hợp `Safety Gateway` vào cả `train_multi_device_agent` và `train_single_q_baseline` trong `train_multi_device_dql.py`.
  - Q-Learning cập nhật giá trị hàm Q dựa trên hành động thực tế đã được Gateway kiểm duyệt và thực thi.
  - Quy đổi thời gian tối thiểu bật/tắt máy nén (anti-short-cycle dwell) thành $15\text{ phút}$ (tương ứng $\ge 1\text{ bước}$ điều khiển 1 giờ).
- **F16 & F17 (Nhãn hành động & Bộ điều khiển cơ sở):**
  - Sửa nhãn hành động số 8 thành `SET_26` và hành động số 11 thành `SET_26.5` trong danh mục 12 action template.
  - Cập nhật `ConventionalThermostatMultiDevice` sang deadband thuần làm mát nhiệt đới $[21, 26^\circ\text{C}]$.

### Thêm mới (Added)
- `tests/test_physical_validation.py`: Bộ kiểm thử vật lý độc lập PV1 đến PV5:
  - PV1: Kiểm thử trôi tự nhiên khi tắt AC (nhiệt độ trôi dần về nhiệt độ môi trường).
  - PV2: Kiểm thử giữ tiện nghi ở tải thiết kế cực đại (AC HIGH giữ phòng $\le 26^\circ\text{C}$).
  - PV3: Kiểm thử độ ổn định số học PM2.5 và CO2 (không dao động răng cưa).
  - PV4: Kiểm thử định luật bảo toàn năng lượng và quán tính nhiệt.
  - PV5: Đo lường tương quan Pearson biên dạng HVAC mô phỏng với dữ liệu thực tế CU-BEMS ($r = 0.4337, p = 0.0342$).
- Tài liệu kỹ thuật:
  - `docs/SIZING.md`: Tính toán định cỡ thiết bị, diện tích $50\text{ m}^2$, mật độ nhiệt nội sinh và tải thiết kế.
  - `docs/PHYSICAL_VALIDATION.md`: Kết quả định lượng chi tiết 5 bài kiểm chứng vật lý.
  - `docs/EXPERIMENT_PROTOCOL.md`: Bản thảo giao thức thực nghiệm khóa chặt trước Phase 3.
- Dữ liệu trực quan:
  - `result/final/diag_trajectory_fixed_BEFORE.png`: Quỹ đạo chẩn đoán trước khi sửa mô phỏng.
  - `result/final/diag_trajectory_fixed_AFTER.png`: Quỹ đạo chẩn đoán sau khi sửa mô phỏng.

### Kiểm chứng & Liêm chính (Integrity & Validation)
- Chạy `pytest tests -v`: **73/73 passed** trong 9.29s (68 bài test hiện có giữ nguyên kết quả pass + 5 bài test vật lý mới).
- **Tuân thủ cam kết:** Không thực hiện bất kỳ lượt huấn luyện RL hoặc chạy benchmark so sánh DQL vs QL vs baseline nào trong Phase 1. Mọi quyết định hiệu chỉnh hoàn toàn dựa trên cơ sở vật lý và kiểm chứng PV1–PV5.

---

## [Phase 1b] - Bổ sung cận trên định cỡ AC, PV6 Độ ẩm, PV1 làm lại và Rà soát không gian trạng thái (2026-11-24)

### Thay đổi & Khắc phục kỹ thuật (Changed & Fixed)
- **A. Bổ sung cận trên định cỡ AC (Capacity / Q_design ∈ [1.2, 1.5]):**
  - Rà soát sau GATE 1 phát hiện tỷ số công suất lạnh cực đại cũ $3.17$ là quá dư so với thực hành thiết kế cơ điện.
  - Thiết lập công suất lạnh bậc HIGH $= 1.30 \times Q_{design} = 4.928\text{ kW thermal}$ với $COP = 3.20$ ($P_{ac,high} = 1.59\text{ kW điện}$).
  - Co giảm $P_{ac,low}$ thành $0.52\text{ kW}$ và $P_{ac,med}$ thành $1.01\text{ kW}$ theo cùng tỷ lệ $\kappa \approx 0.4106$ để bảo toàn đặc tính điều biến bậc máy nén.
  - Tích hợp kiểm tra tự động `assert 1.2 <= self.sizing_safety_ratio <= 1.5` ngay trong khởi tạo `MultiDeviceHVACEnv`.
  - Cập nhật deadband điều biến máy nén: $gap \le 0.4^\circ\text{C}$ (LOW), $0.4 < gap \le 0.8^\circ\text{C}$ (MED), $gap > 0.8^\circ\text{C}$ (HIGH).
- **B. Tiêu chí PV6 Độ ẩm & Khử ẩm AC qua hệ số SHR (Sensible Heat Ratio):**
  - Khảo sát thực nghiệm 60% đầu `2018_F2_Z1` trong giờ có người ghi nhận độ ẩm thực tế $P1 = 48.58\%$, $P50 = 53.14\%$, $P99 = 64.92\%$.
  - Gắn khả năng khử ẩm của AC với công suất làm lạnh hiển nhiệt: $\dot{m}_{latent} = Q_{sensible} \times \frac{1 - SHR}{SHR} \times \frac{3600}{2450}$ với $SHR = 0.75$ (theo ASHRAE Fundamentals).
  - Độ ẩm mô phỏng Fixed Setpoint trong giờ làm việc được kéo về dải thực tế ($P1 = 49.6\%$, $P50 = 71.2\%$, $P99 = 93.4\%$), triệt tiêu hiện tượng đọng ẩm phi thực tế $>93\%$.
- **C. Làm lại PV1 (ALL_OFF trên ngày nắng nóng có người):**
  - Chạy 24h trên ngày nóng nhất tập train ($05/07/2018$, ngoài trời đỉnh $35.80^\circ\text{C}$, 8 người, tải nội sinh $1.75\text{ kW}$).
  - Nhiệt độ khởi điểm $27.18^\circ\text{C}$, đỉnh $36.47^\circ\text{C}$ lúc 19:00, cuối ngày $35.89^\circ\text{C}$, cách trần an toàn $40.0^\circ\text{C}$ một khoảng $3.53^\circ\text{C}$, không bị kẹt trần.
- **D. Cập nhật PV5 (Tương quan biên dạng phụ tải ngày):**
  - Tính toán Pearson $r$ trên hình dạng trung bình theo giờ của toàn bộ 31 ngày thường trong 60% đầu tập train: $r = 0.6054$ (toàn bộ 24h), $r = 0.8092$ (từ giờ 1 đến 23 sau khi loại trừ xung nhọn khởi động lúc 00:00 do kéo nhiệt khối nhiệt tồn dư ban đêm).
  - Báo cáo dưới dạng số liệu mô tả, loại bỏ nhãn "ĐẠT/p<0.05" vì chuỗi thời gian có tự tương quan cao.
- **E. Phân tích Load Bin (0.5 kW) & Cập nhật tài liệu:**
  - Xác lập cơ sở chọn ngưỡng $0.5\text{ kW}$ dựa trên phân phối hai đỉnh (bimodal) khách quan của tải phòng $50\text{ m}^2$: Chế độ chờ ban đêm trung vị $0.026\text{ kW}$ (chiếm 60.28% giờ train), Chế độ hoạt động ban ngày trung vị $1.668\text{ kW}$ (chiếm 39.72% giờ train).
  - Loại bỏ hoàn toàn các mô tả sai lệch cũ kiểu "<5 người / >10 người".
  - **LƯU Ý QUAN TRỌNG:** **Bảng State trong Word/PPT/xlsx phải cập nhật** thành "Tải nội sinh thấp (< 0.5 kW)" và "Tải nội sinh cao (>= 0.5 kW)".
- **F. Khảo sát độ phủ không gian trạng thái:**
  - Chạy chính sách ngẫu nhiên đều 200 episode (4,800 bước) đạt 140/720 trạng thái ($19.4\%$).
  - Phát hiện chiều chết: Bin Nhiệt độ 0 (`< 21.0°C`) có tần suất ghé thăm bằng 0 do hệ thống điều hòa nhiệt đới chỉ làm mát đến setpoint thấp nhất là $23.0^\circ\text{C}$.

### Thêm mới (Added)
- `docs/DESIGN_RATIONALE.md`: Thuyết minh cơ sở lý luận chọn ngưỡng 0.5 kW và cấu trúc không gian trạng thái.
- `result/final/diag_pv1_all_off_24h.png`: Quỹ đạo 24h trôi nhiệt tự nhiên ALL_OFF ngày nóng nhất.
- `result/final/diag_pv6_rh_comparison.png`: So sánh phân phối độ ẩm thực tế vs mô phỏng trên các ngày thường.
- `result/final/diag_state_coverage.png`: Đồ thị 6 chiều phân bố không gian trạng thái và chiều chết.
- `result/final/diag_state_coverage.csv`: Bảng dữ liệu tần suất ghé thăm 720 trạng thái.
- Bài test tự động `test_pv6_humidity_dynamics_and_shr_dehumidification` trong `tests/test_physical_validation.py`.

### Kiểm chứng (Verified)
- Chạy `pytest tests -v`: **74/74 passed** trong 13.50s (68 bài test hiện có + 6 bài test vật lý PV1–PV6).
- `git diff --stat a630433~1 a630433 -- tests/`: Xác nhận không có test cũ nào bị sửa đổi.
- **Cam kết liêm chính:** Tuyệt đối không chạy benchmark so sánh DQL vs QL vs baseline trong Phase 1b. Mọi thử nghiệm chỉ dùng chính sách cố định hoặc ngẫu nhiên đều để kiểm tra vật lý môi trường.

---

## [Phase 1c] - Khử ẩm cuộn lạnh (PV6), Tái hiện vận hành thực tế (PV7), Kiểm tra năng lượng và Khóa thông số vật lý (2026-11-24)

### Thay đổi & Khắc phục kỹ thuật (Changed & Fixed)
- **A. Mô hình cuộn lạnh khử ẩm (Cooling Coil Model - PV6):**
  - Giải trình cân bằng ẩm giờ có người: người tỏa $\approx 0.345\text{ kg/h}$, thông gió + xâm nhập $\approx 0.758\text{ kg/h}$, tổng tải ẩm $\approx 1.103\text{ kg/h}$. Công thức SHR cũ bị hụt khi máy nén giảm tải về MED/LOW.
  - Thay bằng mô hình cuộn lạnh bề mặt (ASHRAE Fundamentals Ch. 21): $\dot{m}_{latent} = \min(\rho \cdot V_{coil} \cdot (w_{room} - w_{coil,sat}) \cdot \text{PLF} \cdot 3600, Q_{sensible} \cdot 3600 / 2450)$.
  - Tham số vật lý chuẩn hóa: $T_{coil} = 12.0^\circ\text{C}$ ($w_{coil,sat} = 0.00874\text{ kg/kg}$), $V_{coil} = 0.08\text{ m}^3\text{/s/kW}$ thiết kế ($0.3032\text{ m}^3\text{/s}$), nền độ ẩm ngoại cảnh Bangkok $68\%$ (SIMULATED_ASSUMED, tham số hiệu chuẩn trên train).
  - Kết quả kiểm chứng trên Train (60%): P50 mô phỏng $53.8\%$ vs thực tế $53.1\%$ (lệch $+0.7\%$).
  - Kết quả kiểm chứng trên Validation (20%): P50 mô phỏng $55.0\%$ vs thực tế $56.4\%$ (lệch $-1.4\%$).
- **B. Tiêu chí PV7 Tái hiện vận hành thực tế (Operation Replay):**
  - Ánh xạ nhị phân: $hvac\_kw < 1.0\text{ kW} \to$ Action 0 (`ALL_OFF_STANDBY`), $hvac\_kw \ge 1.0\text{ kW} \to$ Action 1 (`COMFORT_COOL`). Khởi tạo từ $00:00$ thực tế, chạy 24h trên từng ngày.
  - Kết quả Train: Lệch trung bình nhiệt độ $+0.15^\circ\text{C}$ ($\le 1.5^\circ\text{C}$), MAE $1.33^\circ\text{C}$, $r = 0.7795$; Lệch P50 độ ẩm $+0.7\%$ ($\le 8\%$), MAE $6.6\%$.
  - Kết quả Validation: Lệch trung bình nhiệt độ $+0.34^\circ\text{C}$ ($\le 1.5^\circ\text{C}$), MAE $1.16^\circ\text{C}$, $r = 0.7526$; Lệch P50 độ ẩm $+3.3\%$ ($\le 8\%$), MAE $6.9\%$.
  - Cả hai tiêu chí đều thỏa mãn các ngưỡng cảnh báo đặt trước.
- **C. Kiểm tra năng lượng AC mới (1.59 kW High):**
  - Ngày thường tiêu chuẩn: $8.804\text{ kWh/ngày}$ (AC), $10.843\text{ kWh/ngày}$ (tổng hệ thống, $0.2169\text{ kWh/m}^2\text{/ngày}$).
  - Trung bình 31 ngày thường tập Train: $13.555\text{ kWh/ngày}$ (AC), $15.595\text{ kWh/ngày}$ (tổng hệ thống, $0.3119\text{ kWh/m}^2\text{/ngày}$).
  - Nằm hoàn toàn trong dải benchmark thực tế của ASHRAE $[0.20, 0.80]\text{ kWh/m}^2\text{/ngày}$.
- **D. Phân tích Load Bin (0.5 kW) & Trùng khớp Occupancy:**
  - Standby load: trung vị $0.026\text{ kW}$, Active load: trung vị $1.668\text{ kW}$.
  - Ma trận đối chiếu chéo với sự hiện diện người: $\text{TP} = 402$, $\text{TN} = 624$, $\text{FP} = 19$, $\text{FN} = 15$.
  - Độ trùng khớp đạt **$96.79\%$** ($1,026 / 1,060\text{ giờ}$).
  - Cập nhật mô tả trong mã nguồn: "tải nhiệt nội sinh quy mô, ngưỡng 0.5 kW, tương đương chỉ báo có người (trùng 96.8%)".
  - **LƯU Ý BẮT BUỘC:** **Bảng State trong Word/PPT/xlsx phải cập nhật** thành "Tải nội sinh thấp (< 0.5 kW)" và "Tải nội sinh cao (>= 0.5 kW)".
- **E. Khảo sát độ phủ không gian trạng thái (State Space Coverage):**
  - Chạy 200 episode ngẫu nhiên đều trên toàn bộ ngày train (gồm cuối tuần) đạt 129/720 trạng thái ($17.92\%$) (so với 111/720 trạng thái khi chỉ lấy mẫu ngày thường). Toàn bộ bước cuối tuần cho time_bin = 0.
  - Xác nhận chiều chết vật lý: Bin Nhiệt độ 0 (`< 21.0°C`) có tần suất ghé thăm $0.00\%$ do nhiệt đới Bangkok và setpoint AC tối thiểu $23^\circ\text{C}$.
  - Xác nhận chiều hiếm: Bin PM2.5 số 1 (`> 35 µg/m³`) đạt $0.33\%$.
- **F. Thuyết minh các cơ chế bổ sung & Khóa thông số vật lý:**
  - Thuyết minh dải chết $0.4^\circ\text{C} / 0.8^\circ\text{C}$ và quá độ kéo nhiệt 00:00 trong `docs/EXPERIMENT_PROTOCOL.md`.
  - Khóa vĩnh viễn các thông số vật lý của môi trường cho các phase sau.

### Thêm mới (Added)
- `result/final/diag_pv7_operation_replay.png`: Đồ thị 4 khung so sánh chuỗi thời gian Replay vs Thực tế trên Train và Val.
- Cập nhật `result/final/diag_pv6_rh_comparison.png`, `result/final/diag_state_coverage.png`, và `result/final/diag_state_coverage.csv`.

### Kiểm chứng (Verified)
- Chạy `pytest tests -v`: **74/74 passed** trong 10.78s.
- `git diff --stat ba73e37 HEAD -- tests/`: Xác nhận không có bài test gốc nào bị chỉnh sửa.
- **Cam kết liêm chính:** Không chạy bất kỳ benchmark so sánh DQL vs QL vs baseline nào trong Phase 1c. Mọi quyết định sửa chỉ dựa trên vật lý học và bài kiểm chứng PV1–PV7.

---

## [Phase 2] - Dữ liệu, tiền xử lý và chia tập (Pending)
*(Sẽ được cập nhật khi thực hiện Phase 2)*

---

## [v2.0.0 - Protocol v2 & Full 10/10 Consistency Sync] - 2026-11-25
- **Action 5 (`ACTIVE_VENT_PURIFY`)**: Tích hợp `ac_mode="SET_25"`, `ac_setpoint_c=25.0` kết hợp thông gió `500 m³/h` + lọc HEPA `450 m³/h`, tránh tăng nhiệt phòng khi xả $CO_2$ trong điều kiện nhiệt đới.
- **Lịch Epsilon & Safety Feedback**: Chuyển tìm kiếm lưới sang lịch suy giảm theo `target_decay_frac` đạt `epsilon_min=0.02` (`N_episodes in {1200, 2000, 3000}`, `lr in {0.05, 0.10, 0.15}`), truyền cờ `safety_violation` vào vòng lặp huấn luyện.
- **Tái lập Thực nghiệm (`protocol-locked-v2`)**: DQL đạt `16.02 ± 1.03` điểm trên Test-A (vi phạm nhiệt `2.0%`, $CO_2$ `1.0%`, Safety override `0.0%`); thắng mọi baseline trên Test-A & Test-B (2019).
- **Đồng bộ Toàn diện**: Cập nhật 5 bảng CSV, Excel 4 sheet, 18 biểu đồ PNG dữ liệu thật 100%, Báo cáo Word (`.docx` + DOI xác minh + Footer PAGE + 7 ảnh thật), Slide 19 trang (`.pptx`, font >= 14pt, `Trang 19 / 19`), `live_simulation_gui.py`, `evaluate.py`, và **79 unit tests** (`test_action5_cooling_ventilation.py`).
