# KẾ HOẠCH PHỐI HỢP NHÓM & PHÂN CÔNG TRÁCH NHIỆM (TEAM PLAN)

Đồ án: **"Smart HVAC Control by Double Q-Learning"** (Hướng A)  
Đơn vị: Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)  
Khoa: Công nghệ Thông tin  
Giảng viên hướng dẫn: TS. Nguyễn Thanh Long  
Nhóm sinh viên thực hiện: Nhóm 2 (Đóng góp đồng đều 25% mỗi thành viên)

---

## 1. Bảng phân công chi tiết 4 thành viên

| STT | Thành viên & MSSV | Vai trò chính | Tỷ lệ đóng góp | Phase phụ trách | Chương mục trong Báo cáo Word | Slide thuyết trình (Làm & Nói) | Trọng tâm phản biện cần nắm vững |
|:---:|---|---|:---:|:---:|---|:---:|---|
| 1 | **Nguyễn Vũ Đức Anh**<br>MSSV: 2001240016 | Trưởng nhóm, kiến trúc dữ liệu, tích hợp & QA tổng thể | 25% | **Phase 0, 2**<br>Tích hợp Phase 4, 5, 6<br>QA tổng | Bìa, Lời cam đoan, Ch.1, Ch.3, Ch.8, Tài liệu tham khảo, Mục lục | **Slide 1–5, 9, 18**<br>(Mở đầu, dữ liệu, tiền xử lý, reward, kết luận) | Nguồn gốc dữ liệu CU-BEMS, đường ống tiền xử lý 1h / nội suy $\le$ 3h, thiết kế hàm thưởng 7 thành phần, đóng góp tổng thể của đồ án. |
| 2 | **Trần Thị Thanh Ngân**<br>MSSV: 2001240292 | Kỹ sư thuật toán RL & Phân tích lý thuyết | 25% | Thuật toán, Siêu tham số, Thí nghiệm Maximization Bias, Q-table | Ch.2 (mục 2.3–2.6), Ch.5.3–5.4, Ch.6.2, Ch.6.5 | **Slide 7, 8, 10, 11**<br>(Không gian State/Action, cơ chế Double Q vs Q, phân tích Q-table) | Cơ chế khử thiên lệch Maximization Bias của Double Q-Learning, thiết kế 720 state & 12 action, ý nghĩa độ phủ Q-table 16.1%, thí nghiệm toy problem. |
| 3 | **Võ Thanh Trường**<br>MSSV: 2001240529 | Kỹ sư vật lý nhiệt, Safety Gateway & Giao diện Control Room | 25% | **Phase 1, 7**<br>(Mô hình nhiệt vật lý, Gateway, GUI & Demo) | Ch.4.1–4.2, 4.6–4.7, Ch.5.1–5.2, 5.5–5.6, Phụ lục A | **Slide 12, 16, 17**<br>(Digital Twin, Safety Gateway, Demo 3 màn, Hạn chế thực tế) | Cân bằng nhiệt phòng và định cỡ thiết bị, cơ chế 2 thang thời gian (1h điều khiển / 5 phút tích phân), luật can thiệp an toàn Gateway, vận hành GUI Control Room. |
| 4 | **Võ Thị Diễm My**<br>MSSV: 2001240280 | Kỹ sư thực nghiệm, Thống kê & Đảm bảo chất lượng (QA) | 25% | **Phase 3, 8**<br>(Giao thức thực nghiệm, chạy benchmark, kiểm toán) | Ch.6.1, 6.3, 6.4, 6.6–6.9, Ch.7, Phụ lục B, D | **Slide 6, 13, 14, 15**<br>(Chia tập train/val/test, 3 tiêu chí, DQL vs QL thống kê, Pareto) | Giao thức khóa protocol trước khi chạy, kiểm định thống kê cặp (Paired t-test, Wilcoxon, Cohen's d), đánh đổi Pareto giữa điện năng và tiện nghi, độ bền trên miền trôi dạt. |

---

## 2. Quy tắc phối hợp & Duyệt chéo bắt buộc (Cross-Review)

1. **Duyệt chéo giữa các cặp chuyên môn**:
   - **Võ Thanh Trường $\longleftrightarrow$ Võ Thị Diễm My** (Phase 1): Diễm My kiểm tra tính chặt chẽ của các bài test kiểm chứng vật lý (`test_physical_validation.py`) trước khi chốt mô phỏng của Thanh Trường.
   - **Nguyễn Vũ Đức Anh $\longleftrightarrow$ Trần Thị Thanh Ngân** (Phase 2): Thanh Ngân kiểm tra xem cách chia tập (train/val/test-A/test-B) có gây rò rỉ dữ liệu hoặc sai lệch phân phối cho thuật toán RL hay không.
   - **Trần Thị Thanh Ngân $\longleftrightarrow$ Võ Thị Diễm My** (Phase 3): Diễm My kiểm tra việc cố định seed, không gian siêu tham số và luật kiểm định thống kê của Thanh Ngân trước khi bấm chạy thực nghiệm lớn.
   - **Cả 4 thành viên cùng duyệt**: Toàn bộ báo cáo Word (`.docx`) và slide thuyết trình (`.pptx`) bản cuối cùng.

2. **Quy tắc làm chủ mã nguồn**:
   - Từng thành viên bắt buộc phải tự tay chạy các script thuộc phần mình phụ trách trên máy cá nhân, hiểu rõ từng tham số và giải thích được mọi biểu đồ xuất hiện trong slide của mình.

3. **Lịch trình thực hiện đề xuất (7 ngày)**:
   - **Ngày 1**: Hoàn thành Phase 0 (Đóng băng hiện trạng) & Phase 1 (Sửa mô phỏng đúng vật lý).
   - **Ngày 2**: Hoành thành Phase 2 (Dữ liệu, tiền xử lý, chia tập) & Viết giao thức thực nghiệm khóa (`protocol-locked`).
   - **Ngày 3**: Hoàn thành Phase 3 (Chạy tìm siêu tham số, chạy 10 seed kiểm định thống kê, xuất `manifest.json`).
   - **Ngày 4**: Hoàn thành Phase 4 (Sinh tự động toàn bộ bảng thiết kế và 18 biểu đồ/hình vẽ chuẩn học thuật).
   - **Ngày 5**: Hoàn thành Phase 5 (Soạn thảo và cập nhật báo cáo Word đầy đủ 8 chương theo số liệu từ manifest).
   - **Ngày 6**: Hoàn thành Phase 6 (Xây dựng slide thuyết trình PowerPoint 15 phút + 5 phút Q&A, kịch bản speaker notes, bộ 25 câu hỏi phản biện) & Phase 7 (Kiểm tra GUI/Demo). Tổ chức tập thuyết trình ít nhất 2 lượt toàn nhóm.
   - **Ngày 7**: Hoàn thành Phase 8 (Kiểm toán toàn diện, kiểm tra tính nhất quán Source–Word–PPT, đóng gói nộp bài).
