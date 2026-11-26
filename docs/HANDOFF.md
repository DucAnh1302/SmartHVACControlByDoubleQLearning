# BÀN GIAO DỰ ÁN (PROJECT HANDOFF)

## 1. Nhận diện Dự án & Giao thức
- **Đề tài**: Điều khiển Hệ thống HVAC Đa Thiết bị bằng Thuật toán Double Q-Learning dựa trên Mô hình Digital Twin (Smart HVAC Control).
- **Đơn vị**: Nhóm 2 — Khoa CNTT, ĐH Công Thương TP.HCM (HUIT).
- **Giảng viên hướng dẫn**: **TS. Nguyễn Thanh Long**.
- **Giao thức hiệu chỉnh (Protocol v2)**:
  1. `Action 5 (ACTIVE_VENT_PURIFY)` được trang bị làm mát đồng thời `SET_25` (`25.0°C`) để xả $CO_2$ ($500\text{ m}^3\text{/h}$) và lọc HEPA ($450\text{ m}^3\text{/h}$) không gây quá nhiệt phòng nhiệt đới.
  2. Truyền cờ `safety_violation=(safe_act != raw_action)` vào `MultiDeviceRewardFunction.calculate_reward()` trong cả huấn luyện lẫn đánh giá.
  3. Lịch suy giảm `epsilon` hội tụ sạch về `epsilon_min=0.02` qua `target_decay_frac` trên lưới `N_episodes in {1200, 2000, 3000}` và `lr in {0.05, 0.10, 0.15}`.

## 2. Kết quả Chính & Trạng thái Các Sản phẩm
- **Thực nghiệm chính thức (`result/final/manifest.json`)**:
  - DQL và QL đạt hội tụ tối ưu trên Validation ở `lr=0.15, tf=0.85, episodes=3000, eps_decay=0.998467` (Val Reward DQL `15.23`, QL `15.16`).
  - Trên Test-A: DQL đạt **$16.02 \pm 1.03$** ($20.00 \pm 1.40\text{ kWh/ngày}$, lỗi nhiệt $2.0\%$, lỗi $CO_2$ $1.0\%$, can thiệp an toàn $0.0\%$); QL đạt **$15.42 \pm 2.11$** ($20.73 \pm 4.92\text{ kWh/ngày}$); vượt trội có ý nghĩa thống kê so với Fixed Setpoint ($9.12$), Rule-Based ($8.04$) và Thermostat ($-63.07$).
  - Phân tích DQL vs QL: $t = 0.8312, p = 0.4274 \implies$ **Nhánh diễn giải 2**, DQL ổn định phương sai hơn ~4x so với QL ($1.06$ vs $4.47$) và dẫn đầu trên cả Test-A lẫn Test-B 2019 ($4.94$ vs $2.41$).
- **Bộ sản phẩm đã đồng bộ 100%**:
  - Mã nguồn: `src/` (`action_descriptions_vi.py`), `app/live_simulation_gui.py` (Việt hóa 100%, mở rộng Bảng Bước 12 cột kèm cấu hình Cửa sổ/Thiết bị, tải `final_dql_seed0.pkl`, `final_ql_seed0.pkl`), `evaluate.py`, `run_final_experiments.py`.
  - Bảng biểu & Hình ảnh: `result/final/tables/*.csv` (5 tệp), `result/final/figures/*.png` (19 hình dữ liệu thật `H1-H19` gồm Q-Table thực tế), `doc/assets/` (`hero_smart_hvac_cover.png`, `hvac_problem_evidence.png`, `cubems_raw_excel_preview.png`, `cubems_floor2_transform.png`, `gui_raw_vi.png`, `gui_demo_annotated_vi.png`), `doc/THIET_KE_PHAN_THUONG_VA_HANH_DONG_HVAC.xlsx` (4 sheets).
  - Văn bản học thuật: `doc/BAO_CAO_DO_AN_FINAL.docx` (Caption động `SEQ Hình`/`SEQ Bảng` + `Insert Table of Figures`, đồng bộ câu chuyện SmartHouse Arduino Servo 90 & 5 giải đáp hội đồng gồm Ma trận nhầm lẫn Q&A), `doc/BAO_CAO_THUYET_TRINH_FINAL.pptx` (32 slides hiện đại, mọi chữ >= 20pt, `Trang 32 / 32`, đủ mạch `a -> o`, thêm slide ảnh bìa, bảng thành viên, bằng chứng CU-BEMS/IEA/Harvard, slide trống chèn ảnh Arduino, bảng Raw/Processed CU-BEMS, 720 giỏ đủ tên + ngưỡng giờ; bản backup 27 slides lưu tại `doc/BAO_CAO_THUYET_TRINH_BACKUP_27SLIDES.pptx`).

## 3. Lệnh Xác minh Nhanh
```bash
pytest tests/ -v
python tools/check_consistency.py --check-docx --check-pptx
python tools/run_gate.py 7_8
```
