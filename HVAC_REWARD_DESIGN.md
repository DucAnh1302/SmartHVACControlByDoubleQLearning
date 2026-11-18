# MULTI-OBJECTIVE REWARD DESIGN & SIGNED DECOMPOSITION

> [!IMPORTANT]
> **Tuyên bố Phạm vi Nghiên cứu (`RESEARCH_SIMULATION_ONLY`):**  
> Thiết kế hàm thưởng phân rã này phục vụ mục đích tối ưu hóa Pareto đa mục tiêu trong mô phỏng học tăng cường Digital Twin. Các trọng số thưởng phản ánh sự đánh đổi lý thuyết giữa tiện nghi sinh học và chi phí điện năng, **chưa được hiệu chuẩn theo biểu phí thanh toán thương mại thực tế**.

---

## 1. Nguyên lý Phân rã Cộng Tính Có Dấu (Signed Additive Decomposition)

Hàm thưởng tổng thể $R_{total}$ được phân rã thành **7 thành phần độc lập** có thể đo lường và kiểm chứng đại số:

$$R_{total} = R_{temp} + R_{humidity} + R_{co2} + R_{air\_quality} + R_{energy} + R_{switching} + R_{safety}$$

Mỗi thành phần mang ý nghĩa vật lý rõ ràng và có đơn vị đo lường quy đổi tương thích:

```
                                  R_total
     ┌──────────┬──────────┬──────────┼──────────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼          ▼          ▼          ▼
   R_temp     R_rh       R_co2      R_pm       R_ener    R_switch    R_safe
 (Nhiệt độ) (Độ ẩm)     (Khí CO2)  (Bụi mịn)  (Điện kW) (Độ mòn)   (An toàn)
```

---

## 2. Công thức Chi tiết Của Từng Thành Phần

### 2.1. Thành phần Tiện nghi Nhiệt ($R_{temp}$)
Phản ánh mức độ hài lòng theo dải tiện nghi ASHRAE Standard 55:

$$R_{temp} = \begin{cases} 
+0.50 \cdot M_{occ}, & \text{khi } 24.0^\circ\text{C} \le T \le 26.0^\circ\text{C} \text{ (Vùng lý tưởng)} \\
- w_{temp} \cdot 0.4 \cdot (24.0 - T) \cdot M_{occ}, & \text{khi } 21.0^\circ\text{C} \le T < 24.0^\circ\text{C} \text{ (Hơi lạnh)} \\
- w_{temp} \cdot 0.4 \cdot (T - 26.0) \cdot M_{occ}, & \text{khi } 26.0^\circ\text{C} < T \le 29.0^\circ\text{C} \text{ (Hơi nóng)} \\
- w_{temp} \cdot 2.0 \cdot M_{occ}, & \text{khi } T < 21.0^\circ\text{C} \text{ hoặc } T > 29.0^\circ\text{C} \text{ (Nguy hại nhiệt)}
\end{cases}$$

Trong đó $M_{occ}$ là hệ số nhân thời gian: $M_{occ} = 2.0$ trong giờ làm việc văn phòng, và $M_{occ} = 0.5$ vào ban đêm hoặc cuối tuần.

### 2.2. Thành phần Tiện nghi Độ ẩm Tâm trắc ($R_{humidity}$)
Phòng tránh khô niêm mạc hô hấp ($RH < 40\%$) và ẩm mốc sinh nấm ($RH > 65\%$):

$$R_{humidity} = \begin{cases}
+0.10 \cdot M_{occ}, & \text{khi } 40\% \le RH \le 65\% \\
- w_{humidity} \cdot 0.02 \cdot (40 - RH) \cdot M_{occ}, & \text{khi } RH < 40\% \\
- w_{humidity} \cdot 0.02 \cdot (RH - 65) \cdot M_{occ}, & \text{khi } RH > 65\%
\end{cases}$$

### 2.3. Thành phần Thông gió Dưỡng khí ($R_{co2}$)
Ngăn ngừa buồn ngủ, giảm năng suất làm việc và ngạt khí do tích tụ $CO_2$:

$$R_{co2} = \begin{cases}
+0.20 \cdot M_{occ}, & \text{khi } CO_2 \le 800\text{ ppm} \text{ (Không khí trong lành)} \\
- w_{co2} \cdot 0.001 \cdot (CO_2 - 800) \cdot M_{occ}, & \text{khi } 800 < CO_2 \le 1200\text{ ppm} \\
- w_{co2} \cdot \left[ 0.40 + 0.002 \cdot (CO_2 - 1200) \right] \cdot M_{occ}, & \text{khi } CO_2 > 1200\text{ ppm} \text{ (Bí khí nghiêm trọng)}
\end{cases}$$

### 2.4. Thành phần Lọc Sạch Hạt Bụi mịn ($R_{air\_quality}$)
Đảm bảo sức khỏe đường hô hấp theo tiêu chuẩn WHO:

$$R_{air\_quality} = \begin{cases}
0.0, & \text{khi } PM_{2.5} \le 25\ \mu\text{g/m}^3 \\
- w_{pm25} \cdot 0.02 \cdot (PM_{2.5} - 25) \cdot M_{occ}, & \text{khi } 25 < PM_{2.5} \le 50\ \mu\text{g/m}^3 \\
- w_{pm25} \cdot \left[ 0.50 + 0.03 \cdot (PM_{2.5} - 50) \right] \cdot M_{occ}, & \text{khi } PM_{2.5} > 50\ \mu\text{g/m}^3
\end{cases}$$

### 2.5. Thành phần Tiêu thụ Năng lượng Điện ($R_{energy}$)
Phạt trực tiếp theo tổng công suất điện tiêu thụ tức thời của cả 7 nhóm thiết bị, có nhân hệ số biểu giá điện theo giờ cao điểm (Time-of-Use Tariff):

$$R_{energy} = - w_{energy} \cdot P_{total\_kw} \cdot M_{tariff}$$

Trong đó $M_{tariff} = 1.4$ trong khung giờ cao điểm (09:30–11:30 và 17:00–20:00), $M_{tariff} = 1.0$ cho giờ tiêu chuẩn, và $M_{tariff} = 0.7$ cho giờ thấp điểm ban đêm.

### 2.6. Thành phần Hao mòn Đóng ngắt ($R_{switching}$)
Giảm thiểu chuyển đổi trạng thái liên tục giữa các bước điều khiển:

$$R_{switching} = \begin{cases}
- w_{switch}, & \text{nếu } a_\tau \ne a_{\tau-1} \\
0.0, & \text{nếu giữ nguyên chế độ vận hành}
\end{cases}$$

### 2.7. Thành phần Can thiệp An toàn Khẩn cấp ($R_{safety}$)
Phạt nặng khi Agent ra quyết định nguy hiểm buộc Safety Gateway phải kích hoạt cơ chế Override:

$$R_{safety} = \begin{cases}
- w_{safety}, & \text{nếu có sự can thiệp cưỡng chế từ Safety Gateway} \\
0.0, & \text{nếu hành động đề xuất hợp lệ an toàn}
\end{cases}$$

---

## 3. Bảng Cấu hình Trọng số Theo Hồ sơ Ưu tiên (Preference Profiles)

| Hệ số Trọng số | Cân bằng (`balanced`) | Ưu tiên Tiện nghi (`comfort_priority`) | Tiết kiệm Điện (`energy_priority`) | Ưu tiên IAQ (`air_quality_priority`) |
| :--- | :---: | :---: | :---: | :---: |
| $w_{temp}$ | **1.00** | 2.00 | 0.60 | 0.90 |
| $w_{humidity}$ | **0.25** | 0.40 | 0.15 | 0.25 |
| $w_{co2}$ | **0.30** | 0.35 | 0.20 | 0.70 |
| $w_{pm25}$ | **0.25** | 0.30 | 0.15 | 0.60 |
| $w_{energy}$ | **0.12** | 0.04 | 0.25 | 0.08 |
| $w_{switch}$ | **0.35** | 0.20 | 0.50 | 0.30 |
| $w_{safety}$ | **2.00** | 2.00 | 2.00 | 2.00 |
