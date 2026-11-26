# QUY TRÌNH XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU (DATA PIPELINE)

Tài liệu này đặc tả toàn bộ quy trình tiền xử lý, làm sạch, kỹ thuật đặc trưng, phân tách tập dữ liệu và chuẩn hóa dữ liệu cho đồ án "Smart HVAC Control by Double Q-Learning" (Nhóm 2).

---

## 1. Nguồn Dữ liệu Gốc (Raw Sources)

Dữ liệu cảm biến môi trường tòa nhà thực tế gồm 14 file CSV thô được thu thập từ hệ thống quan trắc công trình thông minh tại các khu vực và năm khác nhau:
- **Tập chính (Primary Domain):** `2018_F2_Z1` (Tầng 2, Vùng 1, năm 2018) gồm 44 ngày liên tục quan trắc các đại lượng nhiệt độ phòng, độ ẩm, nồng độ CO2, bụi PM2.5, công suất tiêu thụ HVAC, chiếu sáng, thiết bị và nhiệt độ ngoài trời.
- **Tập mở rộng miền & truyền tải miền (Transfer Domains):**
  - `2018_F1_Z1`: Tầng 1, Vùng 1 (2018).
  - `2018_F2_Z2`: Tầng 2, Vùng 2 (2018).
  - `2018_F3_Z1`: Tầng 3, Vùng 1 (2018).
  - `2019_F2_Z1`: Tầng 2, Vùng 1 (2019) dùng để đánh giá chuyển dịch phân phối theo năm (Test-B).

---

## 2. Tiền Xử lý và Làm sạch Dữ liệu (Preprocessing & Cleaning)

### 2.1. Xử lý khoảng trống dữ liệu (Missing Data Imputation)
- Đối với khoảng thiếu dữ liệu ngắn (khoảng trống $\le 3$ giờ): Áp dụng phương pháp nội suy tuyến tính (linear interpolation) theo chuỗi thời gian để bảo toàn xu hướng dao động liên tục tự nhiên của nhiệt độ và độ ẩm.
- Đối với điểm đầu hoặc điểm cuối: Sử dụng forward-fill hoặc backward-fill có bảo vệ.
- Cột mặt nạ chất lượng dữ liệu (`quality_mask`): Ghi nhận vị trí các bản ghi bị thiếu hoặc ngoại lai nhằm theo dõi mức độ toàn vẹn của chuỗi đo.

### 2.2. Lấy mẫu lại theo chu kỳ 1 giờ (Hourly Resampling)
- Dữ liệu thô ban đầu có tần suất ghi 1 phút hoặc 5 phút. Để phù hợp với chu kỳ điều khiển tác vụ cấp cao của hệ thống HVAC và quy mô mô phỏng RL, toàn bộ chuỗi thời gian được lấy mẫu lại (resample) về bước thời gian đều đặn 1 giờ (`1h`).
- Đại lượng nhiệt độ, độ ẩm, CO2, PM2.5 được lấy trung bình trong khoảng giờ (`mean`), trong khi điện năng tiêu thụ được tích phân tương ứng.

---

## 3. Kỹ thuật Đặc trưng (Feature Engineering)

Các đặc trưng đầu vào phục vụ mô phỏng vật lý và không gian trạng thái RL bao gồm:
1. **Nhiệt độ ngoài trời ($T_{out}$):** Trích xuất từ quan trắc thực tế của trạm khí tượng/cảm biến ngoài trời của tòa nhà.
2. **Độ ẩm ngoài trời giả định ($RH_{out}$):** Sinh tất định từ mã nguồn `domain_loader.py` với tham số hiệu chuẩn cố định `base_ambient_rh = 68.0%` (được xác thực trên tập Train ở Phase 1c).
3. **Tải nhiệt nội sinh (`internal_heat_kw`):** Tính toán tổng hợp từ công suất thiết bị cắm điện (`plug_load`), chiếu sáng (`lighting_load`) và nhiệt lượng phát ra từ người sử dụng phòng.
4. **Ước tính số người hiện diện (`occupancy_est`):** Tính toán từ ngưỡng công suất hoạt động kết hợp động học gia tăng CO2. Thuộc tính này mang chủ yếu thông tin có người/không có người (trùng 96.8% với trạng thái hoạt động thực tế).

---

## 4. Phân tách Tập Dữ liệu Thời gian (Temporal Split) & Vùng đệm Chống rò rỉ (Buffer Isolation)

Khác với học có giám sát trên dữ liệu độc lập (i.i.d.), dữ liệu chuỗi thời gian tòa nhà có tính tự tương quan (autocorrelation) rất cao giữa các giờ kế cận. Do đó:
- **Tỷ lệ phân chia:**
  - **Train:** 60% dữ liệu đầu theo trục thời gian.
  - **Validation:** 20% dữ liệu kế tiếp.
  - **Test-A:** 20% dữ liệu cuối.
- **Vùng đệm cách ly (Temporal Buffer Zone):** Giữa Train và Validation, cũng như giữa Validation và Test-A, hệ thống loại bỏ một khoảng đệm an toàn **24 giờ (1 ngày)**.
  - Vùng đệm này triệt tiêu hoàn toàn khả năng rò rỉ thông tin trạng thái trễ (thermal inertia) giữa các tập.
  - `tools/run_gate.py` tự động kiểm tra khoảng cách thời gian giữa các tập, đảm bảo buffer $\ge 24.0$ giờ và không có bất kỳ điểm dữ liệu nào bị chồng lấn (`assert_no_temporal_overlap: true`).

---

## 5. Chuẩn hóa Đặc trưng Tách biệt (Train-Only Normalization)

Để ngăn ngừa triệt để hiện tượng Data Snooping:
- Bộ chuẩn hóa `TemporalFeatureScaler` (`src/data/domain_loader.py`) chỉ tính toán các thống kê (Mean, Std, Min, Max) độc quyền trên tập **Train**.
- Tham số chuẩn hóa được lưu trữ và áp dụng nguyên trạng để biến đổi tập **Validation** và tập **Test-A / Test-B / Test-C**.
- Tuyệt đối không cho phép sử dụng thống kê gộp toàn bộ dữ liệu (global stats) trong quá trình huấn luyện mô hình.

---

## 6. Kiểm tra Tái lập Dữ liệu Thô vs Dữ liệu Tiền xử lý (Reproducibility Validation)

Pipeline tiền xử lý được tích hợp kiểm tra tự động (`tests/test_phase2_data_pipeline.py`):
- Hệ thống tải lại dữ liệu thô từ 14 file CSV, thực thi tuần tự pipeline tiền xử lý và so sánh trực tiếp với file lưu trữ sẵn `data/processed/2018_F2_Z1_1h.csv`.
- Sai lệch số lượng dòng (`row_diff_pct`), sai lệch trung bình nhiệt độ (`temp_mean_diff_pct`) và độ ẩm (`rh_mean_diff_pct`) bắt buộc phải $\le 1.0\%$ (thực tế kiểm tra đạt $0.00\%$).
