# MA TRẬN ĐỐI SOÁT NHẤT QUÁN TOÀN HỆ THỐNG (FINAL CONSISTENCY MATRIX)

| Hạng mục / Chỉ số | Nguồn chuẩn (Code / Manifest) | Báo cáo Word (.docx) | Slide Thuyết trình (32 Slides >=20pt) | Excel / CSV / Docs | Trạng thái |
|---|---|---|---|---|---|
| **Giảng viên hướng dẫn** | `TS. Nguyễn Thanh Long` (`project_meta.yaml`) | `TS. Nguyễn Thanh Long` (P010) | `TS. Nguyễn Thanh Long` (S01, S02, S32, Notes) | `TS. Nguyễn Thanh Long` | **MATCH** |
| **Mô hình Nhiệt Động học** | `2R-1C` (`140.0 m³`, `C=11,000 kJ/K`) | `2R-1C`, `140.0 m³` (Ch.1, 4, Bảng A.1) | `2R-1C`, `140 m³` (S01, S11, Notes) | `2R-1C`, `140.0 m³` | **MATCH** |
| **Sizing & Định cỡ AC** | `1.3004` (`Capacity/Q_design`) | `1.3004` (Ch.2, Bảng A.1) | Định cỡ 50m² chuẩn hóa (S09, S11, Notes) | `1.3004` (`GATE 1c PASS`) | **MATCH** |
| **DQL Test-A (10 seeds)** | `Reward 16.02 ± 1.03` \| `E 20.00 ± 1.40` | `16.02 ± 1.03` \| `20.00 ± 1.40` (Bảng 6.1) | `16.02 ± 1.03` \| `20.00` kWh (S24, S25, S29) | `manifest.json` \| `STATE.md` | **MATCH** |
| **QL Test-A (10 seeds)** | `Reward 15.42 ± 2.11` \| `E 20.73 ± 4.92` | `15.42 ± 2.11` \| `20.73 ± 4.92` (Bảng 6.1) | `15.42 ± 2.11` \| `20.73` kWh (S24, S25) | `manifest.json` \| `STATE.md` | **MATCH** |
| **Test-A Baselines (CO2%)** | `Fixed 74.4%` \| `Rule 61.7%` \| `DQL 1.0%` | Bảng 6.1 & mục 6.3 chính xác 100% | S21, S24, S29 & Notes chính xác 100% | `benchmark_results.csv` | **MATCH** |
| **Kiểm định cặp & Nhánh** | `t = 0.8312, p = 0.4274` -> **Nhánh 2** | `t = 0.8312, p = 0.4274`, Nhánh 2 (Bảng 6.3) | `p = 0.4274`, Nhánh 2 (S25) | `H14` & `STATE.md` | **MATCH** |
| **Toy Maximization Bias** | `QL 38.09%` vs `DQL 12.32%` | `38.09%` vs `12.32%` (mục 6.2 & Bảng 6.3) | `38.09%` vs `12.32%` (S23 & H15) | `maxbias_toy_curves.csv` | **MATCH** |
| **Kiểm thử Tự động Pytest** | `84 tests` (`17 files`, 0 fail) | `79+ tests` & Phụ lục D cập nhật `84/84 PASSED` | `100% Đạt Gate & Pytest` (S29) | `Gate 1c..7_8 PASS` | **MATCH** |
| **Bảng CSV, Hình, Word TOF** | `19 PNG` + `5 CSV` + `4 Sheets XLSX` | `Caption SEQ Hình/Bảng` + `Insert Table of Figures` + `10 Hình, 17 Bảng` | `32 Slides >=20pt` (`BAO_CAO_THUYET_TRINH_BACKUP_27SLIDES.pptx` backup), ảnh H5/H9/H11-H19 & `doc/assets/` | `check_consistency.py PASS` | **MATCH** |
