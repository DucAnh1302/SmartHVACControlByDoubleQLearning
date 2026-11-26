# TRẠNG THÁI HIỆN TẠI CỦA DỰ ÁN (PROJECT STATE)

- **Cập nhật lần cuối**: 2026-11-25 (Sau hoàn tất Protocol v2 & Kiểm định lại toàn bộ Gate 1c -> 7_8)
- **Phase hiện tại**: ĐÃ HOÀN TẤT TOÀN BỘ 8 PHASES (PROTOCOL v2 + ĐỒNG BỘ CHỐNG SAI LỆCH 100%)
- **Giảng viên hướng dẫn**: **TS. Nguyễn Thanh Long**
- **Commit khóa Giao thức v2**: Tag `protocol-locked-v2` (trước khi chạy lại tập Test)
- **Tag khóa Ngưỡng Gate**: Tag `gates-locked-v2` (ủy quyền người dùng nâng cỡ chữ tối thiểu Slide lên `>= 20pt`) — SHA256 của `tools/gates/gate_config.yaml`:
  `52932b0bf10ac8a972d7f4203937fbdbdcb973c24e05c2d91890597cf6a31a21`

---

## Tóm tắt Các Khóa Vật lý & Kỹ thuật
1. **Không gian Phòng & Định cỡ**: `50.0 m² x 2.8 m = 140.0 m³`, `C_room = 11,000 kJ/K`, `Capacity / Q_design = 1.3004`.
2. **Không gian Trạng thái & Hành động (Protocol v2)**: `720 bins` (`5 x 3 x 3 x 2 x 2 x 4`, tải nội sinh `0.50 kW` trùng `96.8%` trạng thái có người) x `12 templates`. Action 5 (`ACTIVE_VENT_PURIFY`) tích hợp làm mát `SET_25` (`25.0°C`) song song thông gió `500 m³/h` + lọc HEPA `450 m³/h`.
3. **Lịch Epsilon Hội tụ**: `epsilon_min = 0.02`, tỷ lệ phân rã mục tiêu `tf = 0.85`, `N_episodes = 3000`, `lr = 0.15`, `epsilon_decay = 0.998467` (tuyển chọn từ lưới 27 cấu hình x 3 seeds trên Validation).
4. **Kiểm thử Tự động**: **84 unit tests** trên 17 tệp chuyên biệt trong `tests/` (`100% PASSED`, không sửa test cũ).
5. **Tài liệu & Slide Thuyết trình**: Báo cáo Word `doc/BAO_CAO_DO_AN_FINAL.docx` (chuẩn `Caption`/`Insert Table of Figures`) và Slide `doc/BAO_CAO_THUYET_TRINH_FINAL.pptx` (**32 slides**, mọi chữ `>= 20pt`, giữ bản backup 27 slides tại `doc/BAO_CAO_THUYET_TRINH_BACKUP_27SLIDES.pptx`).

---

## Số liệu Thực nghiệm Chính thức (Test-A, 10 Random Seeds x 8 Episodes)

| Bộ điều khiển | Test-A Reward | Test-A Điện năng (kWh/ngày) | Vi phạm Nhiệt (%) | Vi phạm CO2 (%) | Can thiệp Safety (%) |
|---|---|---|---|---|---|
| **Double Q-Learning** | $16.02 \pm 1.03$ | $20.00 \pm 1.40$ | $2.0\% \pm 2.1\%$ | $1.0\% \pm 1.0\%$ | $0.0\% \pm 0.0\%$ |
| **Single Q-Learning** | $15.42 \pm 2.11$ | $20.73 \pm 4.92$ | $3.1\% \pm 3.4\%$ | $2.5\% \pm 1.9\%$ | $0.0\% \pm 0.0\%$ |
| **Fixed Setpoint (25°C)** | $9.12 \pm 0.07$ | $13.21 \pm 0.06$ | $1.1\% \pm 0.0\%$ | $74.4\% \pm 0.6\%$ | $0.0\% \pm 0.0\%$ |
| **Rule-Based Smart HVAC** | $8.04 \pm 0.11$ | $15.26 \pm 0.07$ | $3.3\% \pm 0.4\%$ | $61.7\% \pm 0.8\%$ | $0.0\% \pm 0.0\%$ |
| **Conventional Thermostat** | $-63.07 \pm 2.50$ | $11.57 \pm 0.12$ | $48.4\% \pm 0.8\%$ | $63.1\% \pm 0.6\%$ | $10.4\% \pm 0.3\%$ |
| **Random Policy** | $-35.82 \pm 10.36$ | $16.42 \pm 0.91$ | $67.3\% \pm 4.9\%$ | $51.8\% \pm 6.5\%$ | $5.0\% \pm 1.7\%$ |

- **Chuyển giao miền Test-B (2019, 6,964h)**: DQL dẫn đầu với Reward $4.94 \pm 1.56$ ($14.41\text{ kWh/ngày}$), vượt QL ($2.41 \pm 3.34$), Fixed Setpoint ($2.92 \pm 0.22$) và Rule-Based ($1.64 \pm 0.31$).
- **Kiểm định cặp DQL vs QL (Test-A)**: Paired $t = 0.8312, p = 0.4274$; Wilcoxon $W = 25.0, p = 0.8457$; Cohen's $d_z = 0.2628$; Bootstrap 95% CI $[-0.552, +2.130]$; Levene $p = 0.4798$ (**Nhánh diễn giải 2**).
- **Maximization Bias**: Toy MDP: QL $38.09\%$ vs DQL $12.32\%$; HVAC MC Bias: QL $-1.96$ vs DQL $-4.03$.
