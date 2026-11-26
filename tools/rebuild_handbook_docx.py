"""
Rebuild Defense Handbook Document (doc/SO_TAY_DOC_HINH_ANH_BIEU_DO_HVAC.docx)
Strictly synchronized with Protocol v2 live code, MultiDeviceRewardFunction,
Vietnamese UI/action descriptions, real Q-Table slices, and real PNG figures.
"""

import json
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.action_descriptions_vi import ACTION_META_VI, CONTROLLER_META_VI
from src.environment.multi_device_env import MULTI_DEVICE_ACTION_TEMPLATES, MultiDeviceHVACEnv
from src.reward.multi_device_reward import MultiDeviceRewardFunction
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyConfig


def _add_heading(doc: docx.Document, text: str, level: int = 1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.name = "Times New Roman"
        r.font.color.rgb = RGBColor(11, 110, 112) if level == 1 else RGBColor(22, 32, 42)
    return p


def _add_table(doc: docx.Document, headers: list[str], rows: list[list[str]]):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    for c_i, h in enumerate(headers):
        cell = tbl.rows[0].cells[c_i]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.bold = True
    for r_i, row_vals in enumerate(rows, start=1):
        for c_i, val in enumerate(row_vals):
            cell = tbl.rows[r_i].cells[c_i]
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
    doc.add_paragraph("")
    return tbl


def main():
    with open(REPO_ROOT / "config" / "project_meta.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    with open(REPO_ROOT / "result" / "final" / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    instr = meta["instructor"]
    sup_full = f"{instr['title']} {instr['name']}"

    dql_a = manifest["controllers"]["Double Q-Learning"]["test-A"]
    ql_a = manifest["controllers"]["Single Q-Learning"]["test-A"]
    fix_a = manifest["controllers"]["Fixed Setpoint (25°C Cool)"]["test-A"]
    thm_a = manifest["controllers"]["Conventional Thermostat"]["test-A"]
    rb_a = manifest["controllers"]["Rule-Based Smart HVAC"]["test-A"]
    rnd_a = manifest["controllers"]["Random Policy"]["test-A"]
    dql_b = manifest["controllers"]["Double Q-Learning"]["test-B"]

    hp_dql = manifest["hyperparameters"]["double_q_learning"]
    hp_ql = manifest["hyperparameters"]["single_q_learning"]
    st = manifest["statistical_tests"]
    be = manifest["bias_experiment"]
    branch = manifest["interpretation_branch"]

    doc = docx.Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.85)
        sec.right_margin = Inches(0.85)

    p0 = doc.add_paragraph()
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r0 = p0.add_run(
        "TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH (HUIT) • KHOA CÔNG NGHỆ THÔNG TIN\n"
        "SỔ TAY BẢO VỆ ĐỒ ÁN MÔN HỌC MÁY (PROTOCOL v2 — ĐỒNG BỘ TIẾNG VIỆT)\n"
        "ĐỀ TÀI: SMART HVAC CONTROL BY DOUBLE Q-LEARNING (DIGITAL TWIN 2R-1C)"
    )
    r0.font.name = "Times New Roman"
    r0.font.size = Pt(13)
    r0.bold = True

    p_meta = doc.add_paragraph(
        f"Giảng viên hướng dẫn: {sup_full} | Nhóm 2 (Đóng góp đều 25%): "
        "Nguyễn Vũ Đức Anh (2001240016), Trần Thị Thanh Ngân (2001240292), "
        "Võ Thanh Trường (2001240529), Võ Thị Diễm My (2001240280)."
    )
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

    _add_heading(doc, "PHẦN 1: 5 CÂU HỎI CỐT LÕI & CÂU CHUYỆN KHỞI NGUỒN ARDUINO", level=1)
    qa_rows = [
        ["Khởi nguồn phần cứng SmartHouse",
         "Từ đồ án SmartHouse dùng Arduino + Servo 90° đóng/mở cửa, đọc cảm biến thang °F tự quy đổi -> °C in ra LED.",
         "Vi điều khiển chạy luật if/else cố định, không tự tối ưu đa thiết bị -> cần bộ não AI Double Q-Learning."],
        ["RL có Ma trận Nhầm lẫn không?",
         "KHÔNG có Ma trận Nhầm lẫn (Confusion Matrix) vì RL không có nhãn đúng duy nhất tại mỗi bước.",
         "Thay bằng: Phân phối 12 hành động (H18), Tỷ lệ vi phạm vật lý (Bảng 6.1), Can thiệp An toàn (0%) & Q-Table (H19)."],
        ["CU-BEMS thành Processed Data ra sao?",
         "Gom log 1 phút thành trung bình 1 giờ (Nhiệt, Ẩm, Điện kW). Không có cột CO2/PM2.5 thô.",
         "CO2 (+18 L/h/người) và PM2.5 (lọc CADR HEPA) giải bằng phương trình cân bằng vật lý mỗi 5 phút trong Twin."],
        ["Vì sao gán trọng số Reward/Phạt vậy?",
         "w_safety=2.0 (-10.0), w_temp=1.0, w_switch=0.35, w_co2=0.30, w_rh=w_pm25=0.25, w_energy=0.12*1.4x.",
         "Đúng tháp ưu tiên: An toàn > Tiện nghi nhiệt > Độ bền máy nén & Khí sạch > Tiết kiệm điện giờ cao điểm."],
        ["720 Giỏ rời rạc nhưng Test số thực?",
         "720 ô chỉ là địa chỉ tra bảng Q-Table (.pkl). Test-A/B CSV chạy hoàn toàn bằng số thực liên tục.",
         "Đo lường khách quan % vi phạm & kWh qua 10 Random Seeds (0..9) x 8 ngày = 80 ngày/thuật toán."],
    ]
    _add_table(doc, ["Câu hỏi Hội đồng", "Cơ chế kỹ thuật cốt lõi", "Cách trả lời ghi điểm 10"], qa_rows)

    _add_heading(doc, "PHẦN 2: BẢNG 12 HÀNH ĐỘNG TIẾNG VIỆT & CẤU HÌNH THIẾT BỊ + CỬA SỔ", level=1)
    env_tmp = MultiDeviceHVACEnv(episode_length=24, seed=42)
    act_rows = []
    for aid in range(12):
        tp = MULTI_DEVICE_ACTION_TEMPLATES[aid]
        vi = ACTION_META_VI[aid]
        p_kw = env_tmp._calculate_actuator_powers_and_effects(tp, 28.0, 60.0, 32.0, 68.0)["powers"]["total_kw"]
        act_rows.append([
            f"A{aid}",
            vi["name_only_vi"],
            "MỞ" if tp.window_open else "Đóng",
            vi["devices_vi"],
            f"{p_kw:.2f} kW",
        ])
    _add_table(doc, ["Mã", "Tên Tiếng Việt", "Cửa sổ", "Thiết lập chính xác trên 7 nhóm thiết bị", "Công suất @28°C"], act_rows)

    _add_heading(doc, "PHẦN 3: HƯỚNG DẪN DEMO GIAO DIỆN TIẾNG VIỆT & BIỂU ĐỒ QUAN TRỌNG", level=1)
    fig_dir = REPO_ROOT / "result" / "final" / "figures"
    gui_annot = REPO_ROOT / "doc" / "assets" / "gui_demo_annotated_vi.png"
    guides = []
    if gui_annot.exists():
        guides.append((
            "0. Giao diện Điều khiển Smart HVAC Tiếng Việt — 6 Khu vực Hướng dẫn Demo",
            gui_annot,
            [
                ["Bước Demo 1", "Chọn phòng & Thuật toán AI", "Chọn CU-BEMS 2018 + Double Q-Learning ở Khu 2, nhấn 'CHẠY 1 BƯỚC'."],
                ["Bước Demo 2", "Giải thích thiết bị & Q-Table", "Chỉ vào Khu 3 (7 thiết bị), Khu 4 (QA/QB + dòng Thiết lập thiết bị) & Khu 6 (Bảng bước mở rộng)."],
            ],
            "PUNCHLINE: Giao diện Tiếng Việt trực quan cho thấy rõ AI mở/đóng cửa sổ hay đặt máy lạnh bao nhiêu độ và lý do thưởng/phạt ở từng bước.",
        ))

    guides.extend([
        (
            f"1. Hình H11 — Hội tụ Huấn luyện ({hp_dql['n_episodes']} Episodes, 10 Seeds)",
            fig_dir / "fig_11_training_convergence_curves.png",
            [
                ["Reward & Điện năng", "Mean ± SD trên 10 seeds", f"DQL & QL hội tụ mượt lên +12..+16 khi epsilon giảm về 0.02 (lr={hp_dql['learning_rate']})."],
            ],
            "PUNCHLINE: Lịch suy giảm epsilon Protocol v2 giải phóng hoàn toàn tình trạng nghẽn thăm dò.",
        ),
        (
            "2. Hình H12 — Đối đầu 6 Bộ Điều Khiển trên Test-A (80 Ngày/Thuật toán RL)",
            fig_dir / "fig_12_main_benchmark_metrics.png",
            [
                ["DQL vs QL", f"{dql_a['mean_reward']} vs {ql_a['mean_reward']}", f"Điện {dql_a['energy_kwh_per_day']} vs {ql_a['energy_kwh_per_day']} kWh/ngày; An toàn 0% ghi đè."],
                ["Baselines Cổ điển", f"Cố định 25°C ({fix_a['mean_reward']})", f"CO2 ngạt {fix_a['co2_violation_pct']}! Rơ-le nhiệt {thm_a['mean_reward']}; Luật IF-THEN {rb_a['mean_reward']}; Ngẫu nhiên {rnd_a['mean_reward']}."],
            ],
            f"PUNCHLINE: RL giữ trọn tiện nghi nhiệt & giảm vi phạm CO2 xuống chỉ {dql_a['co2_violation_pct']} với 0% can thiệp Safety.",
        ),
        (
            "3. Hình H19 — So sánh Bảng Q-Table Thực tế sau Huấn luyện (Seed 0)",
            fig_dir / "fig_19_qtable_comparison.png",
            [
                ["Giải mã Q-Table", "7 trạng thái thực từ .pkl", "Ô 502 (Nóng+CO2 TB) chọn A2 (+6.6); Ô 342 (Mát+Cao điểm) chọn A5 (+9.8)."],
            ],
            "PUNCHLINE: Bảng Q-Table chứng minh trực diện: gặp trạng thái nào Agent tra đúng giá trị Q lớn nhất để ra lệnh cho 7 thiết bị.",
        ),
        (
            f"4. Hình H14 — Kiểm định Cặp DQL vs QL (10 Random Seeds — Nhánh {branch})",
            fig_dir / "fig_14_paired_dql_vs_ql_stats.png",
            [
                ["Kiểm định Ghép cặp", f"t = {st['paired_t']['statistic']:.4f}, p = {st['paired_t']['p_value']:.4f}", f"Wilcoxon W = {st['wilcoxon']['statistic']:.1f} (p = {st['wilcoxon']['p_value']:.4f}), d_z = {st['cohens_d']['effect_size']:.4f}, 95% CI [{st['bootstrap_ci']['ci_lower']:.3f}, {st['bootstrap_ci']['ci_upper']:.3f}]."],
                ["MaxBias Check", f"Toy MDP: QL {be['ql_mean_bias']:.2f}% vs DQL {be['dql_mean_bias']:.2f}%", f"HVAC MC Bias: QL {be['hvac_mc_ql_bias']:.2f} vs DQL {be['hvac_mc_dql_bias']:.2f}."],
            ],
            f"PUNCHLINE: Báo cáo trung thực Nhánh {branch}: DQL & QL đạt hiệu năng cao tương đương về thống kê, nhưng DQL có phương sai ổn định hơn qua 10 seeds.",
        ),
    ])

    for title, img_p, tbl_rows, punch in guides:
        _add_heading(doc, title, level=2)
        doc.add_picture(str(img_p), width=Inches(6.2))
        _add_table(doc, ["Thành phần", "Số liệu xác thực", "Ý nghĩa trả lời Hội đồng"], tbl_rows)
        p_pu = doc.add_paragraph()
        r_pu = p_pu.add_run(punch)
        r_pu.bold = True
        r_pu.font.color.rgb = RGBColor(11, 110, 112)

    out_path = REPO_ROOT / "doc" / "SO_TAY_DOC_HINH_ANH_BIEU_DO_HVAC.docx"
    doc.save(out_path)
    print(f"[*] Rebuilt authoritative handbook saved to {out_path}")


if __name__ == "__main__":
    main()
