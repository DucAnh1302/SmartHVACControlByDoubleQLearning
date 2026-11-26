# Điều Khiển Hệ Thống HVAC Đa Thiết Bị Bằng Thuật Toán Double Q-Learning Dựa Trên Mô Hình Digital Twin

**Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) — Khoa Công nghệ Thông tin**  
**Môn học:** Học Máy (Machine Learning) — Học kỳ 4, Năm học 2026  
**Giảng viên hướng dẫn:** TS. Nguyễn Thanh Long  
**Nhóm thực hiện:** Nhóm 2 (Đóng góp đều 25%/thành viên)  
- Nguyễn Vũ Đức Anh — MSSV: 2001240016 (Trưởng nhóm)
- Trần Thị Thanh Ngân — MSSV: 2001240292
- Võ Thanh Trường — MSSV: 2001240529
- Võ Thị Diễm My — MSSV: 2001240280

---

## Tổng quan Kỹ thuật & Kết quả Thực nghiệm (Protocol v2)

Dự án xây dựng môi trường mô phỏng vật lý **Digital Twin 2R-1C** tích hợp psychrometric ẩm Magnus, cân bằng $CO_2$, lọc bụi mịn $PM_{2.5}$ (thể tích phòng $140.0\text{ m}^3$, tỷ số định cỡ $\text{Capacity}/Q_{\text{design}} = 1.3004$) và tầng bảo vệ an toàn chủ động **Hard Safety Gateway (6 luật tất định)**.  
Bằng chứng thực nghiệm qua **10 random seeds độc lập** trên dữ liệu thực **CU-BEMS (2018_F2_Z1 Test-A)** và **chuyển giao miền năm 2019 (Test-B)**:

- **Double Q-Learning (DQL)** đạt Mean Reward **$16.02 \pm 1.03$**, điện năng **$20.00 \pm 1.40\text{ kWh/ngày}$**, vi phạm nhiệt **$2.0\% \pm 2.1\%$**, giảm vi phạm $CO_2$ xuống chỉ **$1.0\% \pm 1.0\%$**, và tỷ lệ can thiệp Safety Gateway **$0.0\%$**.
- **Single Q-Learning (QL)** đạt Mean Reward **$15.42 \pm 2.11$** ($20.73 \pm 4.92\text{ kWh/ngày}$, vi phạm nhiệt $3.1\%$, vi phạm $CO_2$ $2.5\%$).
- **Đối chứng Baselines**: Fixed Setpoint $25^\circ\text{C}$ đạt **$9.12 \pm 0.07$** (nhưng để vi phạm $CO_2$ tới **$74.4\%$**), Rule-Based đạt **$8.04 \pm 0.11$** ($61.7\%$ lỗi $CO_2$), Conventional Thermostat đạt **$-63.07 \pm 2.50$** (Holm-Bonferroni adjusted $p < 10^{-7}$ khi so sánh với DQL).
- **Kiểm định cặp DQL vs QL & Maximization Bias**: Paired $t = 0.8312, p = 0.4274$; Wilcoxon $W = 25.0, p = 0.8457$; Cohen's $d_z = 0.2628$; Bootstrap 95% CI $[-0.552, +2.130]$ (**Nhánh diễn giải 2** — hai thuật toán RL tương đương thống kê ở mức kỳ vọng, DQL có phương sai ổn định hơn ~4x). Thí nghiệm Toy MDP xác nhận DQL giảm chọn nhánh sai từ $38.09\%$ (QL) xuống $12.32\%$.

---

## Cấu trúc Thư mục & Tái lập Toàn phần

- `src/`: Mã nguồn Digital Twin (`multi_device_env.py`), Agent (`multi_device_double_q_agent.py`), Baselines (`multi_device_baselines.py`), Reward 7 thành phần (`multi_device_reward.py`), Safety Gateway (`multi_device_safety_gateway.py`), và bộ nạp CU-BEMS (`domain_loader.py`).
- `run_all.py` & `run_final_experiments.py`: Tái lập toàn bộ Grid Search 27 cấu hình x 3 seeds + 10 seeds Benchmark + kiểm định + Gate 3.
- `run_live_simulation.py` & `app/live_simulation_gui.py`: Giao diện Control Room Tiếng Việt (mở rộng Bảng Bước kèm cấu hình Cửa sổ/Thiết bị) tải checkpoint thật (`final_dql_seed0.pkl`, `final_ql_seed0.pkl`).
- `tools/`: Export bảng CSV/Excel (`export_design_tables.py`), sinh 19 biểu đồ (`generate_figures.py`), đồng bộ Word (`update_word_report.py`, `merge_module_comparison_to_word.py`), sinh asset minh họa (`generate_presentation_assets.py`), xây dựng PowerPoint 32 slides `>= 20pt` (`build_presentation_v2.py`), kiểm tra nhất quán (`check_consistency.py`), và cổng kiểm soát (`run_gate.py`).
- `tests/`: Bộ **84 unit tests** trong 17 tệp chuyên biệt (`pytest tests/ -v`).
- `doc/`: Báo cáo Word (`Nhom2_BaoCao.docx` — gộp đầy đủ mục 6.8 đối chiếu & công thức %) và Slide thuyết trình 32 trang (`Nhom2_BaoCao.pptx` — font `>= 20pt`).
- `data/`: Đã giữ đầy đủ dữ liệu tiền xử lý (`data/processed/`), hiệu chuẩn (`data/calibration/`) và mẫu (`data/sample/`). Bộ dữ liệu thô (`data/raw/*.csv`) và thư mục đầu ra (`result/`) được loại bỏ khỏi repo để nhẹ dung lượng và sẽ tự động sinh lại khi chạy thực nghiệm.

### Khởi chạy Nhanh
```bash
pip install -r requirements.txt
pytest tests/ -v
python evaluate.py
python run_final_experiments.py
python run_live_simulation.py
```
