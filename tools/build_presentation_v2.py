"""
Build Updated Modern Visual Presentation Deck (doc/BAO_CAO_THUYET_TRINH_FINAL.pptx)
Incorporates all user structural refinements:
1. Cover Slide 1 with project title + representative hero visual.
2. Team Members & Primary Roles table right on Slide 2.
3. Empirical HVAC Problem Evidence Slide 3 (CU-BEMS 77.0% AC, IEA >=50%, Harvard COGfx) before Problem Summary Slide 4.
4. Blank Photo Placeholder Slide 5 for user's old Arduino SmartHouse photos before Motivation Slide 6.
5. Removes Confusion Matrix slide from deck (kept in Word/Defense Handbook for Q&A).
6. Raw CU-BEMS Excel Dataset Preview Slide 8 + 2018Floor2 -> Processed Transformation Slide 9 before Data/CO2 Slides 10-11.
7. Ultra-clear 720-State Binning (Slides 13-14) giving BOTH explicit names AND exact numerical/hour thresholds for every single bin.
8. Preserves all validated algorithm, Q-Table, benchmark, GUI demo walkthrough, and closing slides (32 slides total, >=20.0pt everywhere).
"""

import json
import sys
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.action_descriptions_vi import ACTION_META_VI

# Modern Engineering Palette
DARK_BG = RGBColor(15, 23, 42)       # #0F172A Slate-900
LIGHT_BG = RGBColor(248, 250, 252)   # #F8FAFC Slate-50
WHITE = RGBColor(255, 255, 255)
INK = RGBColor(15, 23, 42)
MUTED = RGBColor(71, 85, 105)        # #475569 Slate-600
BORDER = RGBColor(203, 213, 225)     # #CBD5E1
CYAN = RGBColor(8, 145, 178)         # #0891B2 Cyan-600
CYAN_BRIGHT = RGBColor(34, 211, 238) # #22D3EE
CYAN_SOFT = RGBColor(224, 242, 254)  # #E0F2FE
EMERALD = RGBColor(5, 150, 105)      # #059669
EMERALD_SOFT = RGBColor(209, 250, 229)
AMBER = RGBColor(217, 119, 6)        # #D97706
AMBER_SOFT = RGBColor(254, 243, 199)
ROSE = RGBColor(225, 29, 72)         # #E11D48
ROSE_SOFT = RGBColor(255, 228, 230)
INDIGO = RGBColor(79, 70, 229)       # #4F46E5
INDIGO_SOFT = RGBColor(224, 231, 255)

FONT = "Segoe UI"
TOTAL_SLIDES = 32


def _set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_rect(slide, x, y, w, h, fill_rgb: RGBColor, border_rgb: RGBColor = None, bw: float = 1.2):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.04
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_rgb
    if border_rgb is not None:
        shp.line.color.rgb = border_rgb
        shp.line.width = Pt(bw)
    else:
        shp.line.fill.background()
    return shp


def _add_plain_rect(slide, x, y, w, h, fill_rgb: RGBColor):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_rgb
    shp.line.fill.background()
    return shp


def _add_arrow(slide, x, y, w, h, fill_rgb: RGBColor = CYAN):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_rgb
    shp.line.fill.background()
    return shp


def _add_text(
    slide, x, y, w, h, text: str, pt: float = 20.0, bold: bool = False,
    color: RGBColor = INK, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, line_spacing: float = 1.15
):
    assert pt >= 20.0, f"Font violation: {pt}pt < 20pt on text {text[:30]}"
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        if idx > 0:
            p.space_before = Pt(3)
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(pt)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def _add_card(
    slide, x, y, w, h, title: str, body: str, accent: RGBColor = CYAN,
    bg_rgb: RGBColor = WHITE, title_pt: float = 21.0, body_pt: float = 20.0,
    line_spacing: float = 1.15
):
    """Single-TextFrame card: Title + Body flow natively so overlap is impossible."""
    assert title_pt >= 20.0 and body_pt >= 20.0
    _add_rect(slide, x, y, w, h, bg_rgb, border_rgb=BORDER, bw=1.2)
    _add_plain_rect(slide, x + 0.04, y + 0.08, 0.09, h - 0.16, accent)

    tb = slide.shapes.add_textbox(Inches(x + 0.18), Inches(y + 0.08), Inches(w - 0.26), Inches(h - 0.16))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)

    p_t = tf.paragraphs[0]
    p_t.space_after = Pt(6)
    r_t = p_t.add_run()
    r_t.text = title
    r_t.font.name = FONT
    r_t.font.size = Pt(title_pt)
    r_t.font.bold = True
    r_t.font.color.rgb = accent

    for line in body.split("\n"):
        if not line.strip():
            continue
        p_b = tf.add_paragraph()
        p_b.line_spacing = line_spacing
        p_b.space_after = Pt(4)
        r_b = p_b.add_run()
        r_b.text = line
        r_b.font.name = FONT
        r_b.font.size = Pt(body_pt)
        r_b.font.color.rgb = INK


def _add_table(
    slide, x, y, w, h, headers: list[str], rows: list[list[str]],
    col_widths: list[float], pt: float = 20.0, hdr_bg: RGBColor = DARK_BG
):
    assert pt >= 20.0
    tbl_shp = slide.shapes.add_table(1 + len(rows), len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = tbl_shp.table
    for ci, cw in enumerate(col_widths):
        tbl.columns[ci].width = Inches(cw)

    def _fmt(cell, text, is_hdr=False, is_even=False, highlight=False):
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.06)
        cell.margin_right = Inches(0.06)
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.fill.solid()
        if is_hdr:
            cell.fill.fore_color.rgb = hdr_bg
        elif highlight:
            cell.fill.fore_color.rgb = EMERALD_SOFT
        else:
            cell.fill.fore_color.rgb = WHITE if is_even else LIGHT_BG
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if is_hdr else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = str(text)
        r.font.name = FONT
        r.font.size = Pt(pt)
        r.font.bold = is_hdr or highlight
        r.font.color.rgb = WHITE if is_hdr else (EMERALD if highlight else INK)

    for ci, hv in enumerate(headers):
        _fmt(tbl.cell(0, ci), hv, is_hdr=True)
    for ri, rvals in enumerate(rows):
        hi = any(tok in str(rvals[0]) for tok in ["Double Q-Learning", "DQL", "A5", "A4", "Nguyễn Vũ Đức Anh"])
        for ci, val in enumerate(rvals):
            _fmt(tbl.cell(ri + 1, ci), val, is_hdr=False, is_even=(ri % 2 == 0), highlight=(ci == 0 and hi))
    return tbl_shp


def _add_picture_fit(slide, img_path: Path, x: float, y: float, max_w: float, max_h: float):
    with Image.open(img_path) as im:
        pw, ph = im.size
    ratio = pw / ph
    w = max_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    px = x + (max_w - w) / 2.0
    py = y + (max_h - h) / 2.0
    return slide.shapes.add_picture(str(img_path), Inches(px), Inches(py), width=Inches(w), height=Inches(h))


def _chrome(slide, page_num: int, pill: str, title: str, dark: bool = False):
    _set_bg(slide, DARK_BG if dark else LIGHT_BG)
    if not dark:
        _add_plain_rect(slide, 0.0, 0.0, 13.333, 0.09, CYAN)
        pill_w = min(8.6, max(4.6, len(pill) * 0.168 + 0.60))
        _add_rect(slide, 0.45, 0.19, pill_w, 0.44, CYAN_SOFT, border_rgb=CYAN, bw=1.0)
        _add_text(slide, 0.50, 0.19, pill_w - 0.10, 0.44, pill, pt=20.0, bold=True, color=CYAN, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        _add_text(slide, 0.45, 0.67, 12.43, 0.62, title, pt=27.0, bold=True, color=INK)
        _add_plain_rect(slide, 0.45, 1.32, 12.43, 0.025, BORDER)
        footer_col = MUTED
    else:
        footer_col = CYAN_BRIGHT
    _add_text(slide, 10.65, 6.96, 2.25, 0.42, f"Trang {page_num} / {TOTAL_SLIDES}", pt=20.0, bold=True, color=footer_col, align=PP_ALIGN.RIGHT)
    _add_text(slide, 0.45, 6.96, 8.8, 0.42, "Smart HVAC Digital Twin | Nhóm 2 • HUIT", pt=20.0, color=footer_col)


def _notes(slide, txt: str):
    slide.notes_slide.notes_text_frame.text = txt


def main():
    with open(REPO_ROOT / "config" / "project_meta.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    with open(REPO_ROOT / "result" / "final" / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sup_full = f"{meta['instructor']['title']} {meta['instructor']['name']}"
    hp_dql = manifest["hyperparameters"]["double_q_learning"]
    dql_a = manifest["controllers"]["Double Q-Learning"]["test-A"]
    ql_a = manifest["controllers"]["Single Q-Learning"]["test-A"]
    fix_a = manifest["controllers"]["Fixed Setpoint (25°C Cool)"]["test-A"]
    thm_a = manifest["controllers"]["Conventional Thermostat"]["test-A"]
    rb_a = manifest["controllers"]["Rule-Based Smart HVAC"]["test-A"]
    rnd_a = manifest["controllers"]["Random Policy"]["test-A"]
    dql_b = manifest["controllers"]["Double Q-Learning"]["test-B"]
    st = manifest["statistical_tests"]
    be = manifest["bias_experiment"]
    branch = manifest["interpretation_branch"]

    fig_dir = REPO_ROOT / "result" / "final" / "figures"
    assets_dir = REPO_ROOT / "doc" / "assets"

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # --- SLIDE 1: COVER WITH HERO REPRESENTATIVE IMAGE (Yêu cầu 1) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 1, "", "", dark=True)
    _add_plain_rect(s, 0.0, 0.0, 0.22, 7.5, CYAN_BRIGHT)
    _add_rect(s, 0.55, 0.52, 7.05, 0.56, RGBColor(30, 41, 59), border_rgb=CYAN_BRIGHT, bw=1.5)
    _add_text(s, 0.62, 0.54, 6.91, 0.52, "ĐỒ ÁN HỌC MÁY • ĐH CÔNG THƯƠNG (HUIT)", pt=20.0, bold=True, color=CYAN_BRIGHT, valign=MSO_ANCHOR.MIDDLE)
    _add_text(s, 0.55, 1.30, 7.0, 2.05,
              "ĐIỀU KHIỂN SMART HVAC ĐA THIẾT BỊ\nBẰNG DOUBLE Q-LEARNING",
              pt=34.0, bold=True, color=WHITE)
    _add_text(s, 0.55, 3.45, 7.0, 0.90,
              "Từ dự án phần cứng SmartHouse Arduino đến Hệ thống AI Song sinh Số (Digital Twin)",
              pt=21.0, color=CYAN_SOFT)
    _add_rect(s, 0.55, 4.55, 7.0, 1.95, RGBColor(30, 41, 59), border_rgb=CYAN, bw=1.4)
    _add_text(s, 0.72, 4.68, 6.66, 0.52, f"GVHD: {sup_full}", pt=23.0, bold=True, color=CYAN_BRIGHT)
    _add_text(s, 0.72, 5.25, 6.66, 1.15,
              "Nhóm 2 • Khoa Công nghệ Thông tin (HUIT)\n720 Trạng thái × 12 Hành động | 7 Nhóm thiết bị | 10 Seeds",
              pt=20.0, color=WHITE)
    # Right visual hero image representing the system
    _add_picture_fit(s, assets_dir / "hero_smart_hvac_cover.png", 7.75, 0.60, 5.20, 5.95)
    _notes(s, "Kính chào Thầy TS. Nguyễn Thanh Long và Hội đồng. Nhóm 2 xin trình bày đồ án Điều khiển Smart HVAC đa thiết bị bằng Double Q-Learning. "
              "Bên phải trang bìa mô tả kiến trúc tổng quát của đồ án: AI đọc 4 cảm biến phòng qua Digital Twin, xử lý qua 2 bảng QA/QB kèm Cổng An toàn, rồi điều khiển đồng bộ máy lạnh, quạt, cửa sổ và lọc bụi HEPA.")

    # --- SLIDE 2: GIỚI THIỆU THÀNH VIÊN & VAI TRÒ DẠNG BẢNG (Yêu cầu 2) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 2, "GIỚI THIỆU NHÓM 2", "Thành Viên Thực Hiện & Vai Trò Chủ Đạo Trong Đồ Án")
    _add_table(
        s, 0.45, 1.50, 12.43, 4.55,
        ["Thành viên (MSSV)", "Vai trò", "Đóng góp chuyên môn chính", "Tỷ lệ"],
        [
            ["Nguyễn Vũ Đức Anh (2001240016)", "Trưởng nhóm", "Xử lý 14 CSV CU-BEMS, thiết kế hàm thưởng, Word & Slide", "25%"],
            ["Trần Thị Thanh Ngân (2001240292)", "Thành viên", "Thuật toán Double Q-Learning, Q-Table & kiểm chứng Bias", "25%"],
            ["Võ Thanh Trường (2001240529)", "Thành viên", "Mô hình hóa Digital Twin & Giao diện Tiếng Việt (GUI)", "25%"],
            ["Võ Thị Diễm My (2001240280)", "Thành viên", "Benchmark 5 Baselines, kiểm định 10 Seeds & 84 Unit Tests", "25%"],
        ],
        [4.10, 1.75, 5.63, 0.95], pt=20.0
    )
    _add_rect(s, 0.45, 6.15, 12.43, 0.65, CYAN_SOFT, border_rgb=CYAN, bw=1.2)
    _add_text(s, 0.65, 6.18, 12.03, 0.58,
              f"GVHD: {sup_full} — 4 thành viên đóng góp đều 25%, kiểm chứng chéo.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Nhóm 2 gồm 4 thành viên với mức đóng góp đồng đều 25%: bạn Đức Anh phụ trách dữ liệu CU-BEMS và hàm thưởng, bạn Thanh Ngân phụ trách thuật toán Double Q-Learning và kiểm chứng bias, bạn Thanh Trường xây dựng Digital Twin và giao diện GUI, bạn Diễm My thực hiện benchmark các baseline và hệ thống kiểm thử.")

    # --- SLIDE 3: BẰNG CHỨNG THỰC TIỄN VỀ VẤN ĐỀ HVAC (Yêu cầu 3a) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 3, "A1 • BẰNG CHỨNG THỰC TIỄN", "Số Liệu & Nghiên Cứu Khoa Học Chứng Minh Vấn Đề HVAC")
    _add_picture_fit(s, assets_dir / "hvac_problem_evidence.png", 0.45, 1.45, 12.43, 4.30)
    _add_rect(s, 0.45, 5.88, 12.43, 0.92, WHITE, border_rgb=ROSE, bw=1.4)
    _add_text(s, 0.65, 5.95, 12.03, 0.78,
              "Dẫn chứng thật: Trên CU-BEMS Tầng 2 điều hòa chiếm tới 77.0% điện; báo cáo IEA ghi >=50% đỉnh tải; ĐH Harvard chứng minh CO2 >1000 ppm giảm mạnh nhận thức.",
              pt=20.0, bold=True, color=ROSE, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Trước khi trình bày lý do chọn đề tài, nhóm đưa ra 3 bằng chứng thực tiễn cụ thể: đo trực tiếp trên bộ dữ liệu tòa nhà CU-BEMS Tầng 2 cho thấy riêng điều hòa đã chiếm tới 77.0% tổng điện năng; báo cáo IEA ghi nhận HVAC chiếm >=50% phụ tải đỉnh ngày nóng; và nghiên cứu COGfx của ĐH Harvard đăng trên Environmental Health Perspectives chỉ ra nồng độ CO2 945 đến 1400 ppm làm giảm 15% đến 50% điểm nhận thức.")

    # --- SLIDE 4: LÝ DO CHỌN ĐỀ TÀI (Slide 2 cũ - Yêu cầu 3b) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 4, "A2 • LÝ DO CHỌN ĐỀ TÀI", "Bối Cảnh & 3 Hạn Chế Lớn Của Hệ Thống HVAC Cổ Điển")
    _add_card(s, 0.50, 1.55, 3.9, 3.55, "50%–77% Điện Năng",
              "Điều hòa & thông gió chiếm phần lớn điện thương mại; đóng/ngắt cứng gây tốn điện khởi động.",
              accent=ROSE)
    _add_card(s, 4.70, 1.55, 3.9, 3.55, "Mù Khí CO2 & Bụi PM2.5",
              "Rơ-le chỉ nhìn mỗi nhiệt độ; đóng kín phòng bật lạnh 20–22°C gây ngạt CO2 (>1200 ppm) và sốc nhiệt.",
              accent=AMBER)
    _add_card(s, 8.90, 1.55, 3.9, 3.55, "Thiết bị Đánh nhau",
              "Không điều phối đồng thời: mở cửa đón gió nóng lúc máy lạnh đang chạy hoặc tắt lọc bụi lúc ô nhiễm.",
              accent=INDIGO)
    _add_rect(s, 0.50, 5.38, 12.3, 1.25, CYAN_SOFT, border_rgb=CYAN, bw=1.5)
    _add_text(s, 0.70, 5.50, 11.9, 1.0,
              "Đầu vào bài toán: Cần bộ điều khiển tự học biết phối hợp đồng thời Máy lạnh, Quạt gió tươi, Cửa sổ, Hút/Tạo ẩm & Lọc HEPA.",
              pt=21.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Từ các bằng chứng trên, ta thấy rõ 3 vấn đề cốt lõi của điều hòa truyền thống: tiêu thụ phần lớn điện năng tòa nhà, hoàn toàn mù trước khí CO2 và bụi mịn khi đóng kín phòng, và các thiết bị chạy rời rạc không biết phối hợp với nhau.")

    # --- SLIDE 5: SLIDE TRỐNG ĐỂ NGƯỜI DÙNG CHÈN ẢNH ĐỒ ÁN ARDUINO CŨ (Yêu cầu 4a) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 5, "A3 • HÌNH ẢNH MÔ HÌNH CŨ", "Mô Hình Phần Cứng SmartHouse Arduino Đã Thực Hiện")
    placeholder = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(1.55), Inches(12.03), Inches(5.15))
    placeholder.adjustments[0] = 0.02
    placeholder.fill.solid()
    placeholder.fill.fore_color.rgb = WHITE
    placeholder.line.color.rgb = CYAN
    placeholder.line.width = Pt(2.0)
    _add_text(s, 1.2, 3.65, 10.9, 0.85,
              "[ Khung trống để chèn trực tiếp hình ảnh / video mô hình Arduino SmartHouse thực tế của nhóm ]",
              pt=20.0, bold=True, color=MUTED, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Slide này dành riêng để chiếu ảnh thật mô hình SmartHouse Arduino mà nhóm đã chế tạo trước đây: mạch Arduino, động cơ Servo 90 độ đóng/mở cửa, cảm biến nhiệt độ và màn hình LED hiển thị nhiệt độ sau khi quy đổi từ độ F sang độ C.")

    # --- SLIDE 6: TỪ ARDUINO CŨ ĐẾN BỘ NÃO AI (Slide 3 cũ - Yêu cầu 4b) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 6, "A4 • KHỞI NGUỒN BỘ NÃO AI", "Từ Phần Cứng SmartHouse Arduino Đến Bộ Não Double Q-Learning")
    _add_card(s, 0.50, 1.60, 5.4, 3.70, "1. SmartHouse Arduino cũ",
              "• Dùng động cơ Servo 90° đóng/mở cửa tự động.\n"
              "• Cảm biến đọc thang độ Fahrenheit (°F).\n"
              "• Code tự quy đổi °F -> °C và xuất màn hình LED.",
              accent=AMBER, bg_rgb=AMBER_SOFT)
    _add_arrow(s, 6.08, 3.10, 1.15, 0.75, fill_rgb=CYAN)
    _add_card(s, 7.40, 1.60, 5.4, 3.70, "2. Khoảng trống cần Bộ não RL",
              "• Arduino chỉ chạy rẽ nhánh if/else theo ngưỡng cứng (Setpoint).\n"
              "• Không tự giải toán tử tối ưu đa mục tiêu.\n"
              "• Xây SmartHVACControlByDoubleQLearning làm bộ não.",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _add_rect(s, 0.50, 5.52, 12.3, 1.18, WHITE, border_rgb=INDIGO, bw=1.5)
    _add_text(s, 0.70, 5.62, 11.9, 0.98,
              "Chuỗi giá trị: Phần cứng (Arduino/Servo 90° chấp hành lệnh) <--- Bộ não AI (Double Q-Learning tính trước lệnh tối ưu).",
              pt=21.0, bold=True, color=INDIGO, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Khi lập trình trực tiếp trên mô hình Arduino trước đây, nhóm nhận thấy vi điều khiển rất giỏi thực thi lệnh setpoint cố định như if nhiệt độ > X thì quay Servo 90 độ, nhưng không thể tự giải bài toán tối ưu nhiều thiết bị. Do đó đồ án này tập trung xây dựng trước bộ não AI thông minh trước khi kết nối ngược xuống phần cứng.")

    # --- SLIDE 7: VÌ SAO CHỌN RL THAY SL / USL (Đã bỏ Slide Ma trận nhầm lẫn - Yêu cầu 5) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 7, "B • LỰA CHỌN PHƯƠNG PHÁP", "Vì sao dùng Học tăng cường (RL) thay SL hay USL?")
    _add_card(s, 0.50, 1.60, 3.85, 3.70, "Supervised (SL)",
              "Cần bộ nhãn 'đáp án đúng' cho mọi giờ (X -> y). Dữ liệu tòa nhà thực tế KHÔNG có sẵn lệnh điều khiển tối ưu.",
              accent=MUTED)
    _add_card(s, 4.70, 1.60, 3.85, 3.70, "Unsupervised (USL)",
              "Chỉ gom cụm hay tìm mẫu ẩn trong dữ liệu lịch sử; không có cơ chế tương tác thử-sai để điều khiển thiết bị.",
              accent=INDIGO)
    _add_card(s, 8.90, 1.60, 3.90, 3.70, "Reinforcement Learning ✓",
              "Học qua tương tác State -> Action -> Reward. Tối ưu chuỗi điểm thưởng dài hạn 24 giờ cho 7 thiết bị.",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _add_rect(s, 0.50, 5.55, 12.3, 1.12, CYAN_SOFT, border_rgb=CYAN, bw=1.5)
    _add_text(s, 0.70, 5.65, 11.9, 0.92,
              "Kết quả quy trình: RL biến bài toán 'không có nhãn đáp án' thành bài toán ra quyết định chuỗi có kiểm soát an toàn.",
              pt=21.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Supervised Learning cần nhãn đáp án tối ưu từng giờ mà bộ dữ liệu đo đạc không có; Unsupervised chỉ phân cụm chứ không ra quyết định chuỗi. Do hành động giờ này ảnh hưởng nhiệt và CO2 nhiều giờ kế tiếp, Học tăng cường (RL) là giải pháp phù hợp nhất.")

    # --- SLIDE 8: SHOW DẠNG EXCEL BỘ DATASET RAW CU-BEMS (Yêu cầu 6) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 8, "C1 • BỘ DỮ LIỆU GỐC (RAW)", "Quan Sát Bộ Dữ Liệu Thô CU-BEMS (Dạng Bảng Excel 1 Phút)")
    _add_picture_fit(s, assets_dir / "cubems_raw_excel_preview.png", 0.45, 1.45, 12.43, 4.30)
    _add_rect(s, 0.45, 5.88, 12.43, 0.92, AMBER_SOFT, border_rgb=AMBER, bw=1.4)
    _add_text(s, 0.65, 5.95, 12.03, 0.78,
              "Đặc điểm bản gốc chưa tiền xử lý: 14 file CSV (Floor 1–7 năm 2018 & 2019), ghi mỗi 1 phút; có nhiều ô trống NaN và công suất quy mô cả tầng.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Đây là hình ảnh thực tế cấu trúc file CSV thô gốc CU-BEMS khi mở dưới dạng bảng Excel: gồm 14 tệp theo 7 tầng trong 2 năm 2018–2019, tần suất từng phút. Có thể thấy các file gốc xuất hiện nhiều khoảng bị mất tín hiệu cảm biến (ô NaN tô đỏ) và công suất đo cho cả tầng lớn nên chưa thể nạp thẳng cho AI học.")

    # --- SLIDE 9: BIẾN ĐỔI 2018 FLOOR 2 THÀNH PROCESSED DATA (Yêu cầu 7) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 9, "C2 • BIẾN ĐỔI 2018 FLOOR 2", "Quy Trình Tiền Xử Lý: 2018Floor2.csv -> Bộ Processed 1 Giờ")
    _add_picture_fit(s, assets_dir / "cubems_floor2_transform.png", 0.45, 1.45, 12.43, 4.30)
    _add_rect(s, 0.45, 5.88, 12.43, 0.92, EMERALD_SOFT, border_rgb=EMERALD, bw=1.4)
    _add_text(s, 0.65, 5.95, 12.03, 0.78,
              "4 Bước đã làm: Tách Zone 1 Tầng 2 -> Gom trung bình 60 phút thành 1 giờ -> Nội suy khoảng trống <=3h -> Quy đổi tải phòng 50m² & suy người.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Slide này minh họa trực tiếp 2018Floor2.csv được tiền xử lý ra sao thành tệp 2018_F2_Z1_1h.csv (1,767 giờ sạch): nhóm tách riêng Zone 1 Tầng 2, gom trung bình mỗi 60 dòng 1 phút thành 1 bước 1 giờ khớp với chu kỳ điều khiển HVAC, nội suy tuyến tính các đoạn đứt ngắn <= 3 giờ và quy đổi tải về quy mô phòng chuẩn 50m2.")

    # --- SLIDE 10: PHÂN CHIA THỜI GIAN CHỐNG RÒ RỈ ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 10, "C3 • CHIA TẬP DỮ LIỆU", "Nguồn Gốc CU-BEMS & Phân Chia Thời Gian Chống Rò Rỉ")
    _add_picture_fit(s, fig_dir / "fig_05_temporal_split_timeline.png", 0.50, 1.50, 7.6, 4.2)
    _add_card(s, 8.30, 1.55, 4.5, 4.15, "CU-BEMS (Bangkok)",
              "• Tòa văn phòng 7 tầng (F2_Z1 năm 2018 & 2019).\n"
              "• Train 60% | Val 20% | Đệm 25h | Test-A 20%.\n"
              "• Test-B: trọn năm 2019 (6,964 giờ kiểm thử chéo).",
              accent=CYAN)
    _add_rect(s, 0.50, 5.88, 12.3, 0.85, WHITE, border_rgb=EMERALD, bw=1.5)
    _add_text(s, 0.70, 5.95, 11.9, 0.70,
              "Ý nghĩa áp dụng: Chia tuần tự thời gian + vùng đệm 25h ngăn hoàn toàn rò rỉ dữ liệu tương lai khi huấn luyện Q-Table.",
              pt=20.0, bold=True, color=EMERALD, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Từ bộ 2018_F2_Z1_1h.csv đã làm sạch, dữ liệu được chia tuần tự 60% đầu làm tập Train, 20% giữa làm Validation chọn siêu tham số, chừa vùng đệm 25 giờ rồi test trên 20% cuối (Test-A), cùng tập Test-B trọn năm 2019.")

    # --- SLIDE 11: SLIDE 7 CŨ (GOM 1H TẠO CO2/PM2.5 KIỂU GÌ - Yêu cầu 8) ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 11, "C4 • MINH BẠCH NGUỒN BIẾN", "Gom 1 Giờ Từ CU-BEMS — Vậy CO2 & PM2.5 Đến Từ Đâu?")
    _add_card(s, 0.50, 1.55, 3.85, 3.70, "1. CSV Đã Xử Lý (1h)",
              "Chứa giá trị đo thực từ CU-BEMS: Nhiệt, Ẩm, công suất AC, đèn & ổ cắm mỗi 1 giờ [REAL_HISTORICAL].",
              accent=CYAN, bg_rgb=CYAN_SOFT)
    _add_arrow(s, 4.42, 3.05, 0.45, 0.55)
    _add_card(s, 4.95, 1.55, 3.65, 3.70, "2. Suy luận Số người",
              "Đoạn công suất đèn+ổ cắm quy mô >=0.50 kW tương ứng lúc có người làm việc (khớp 96.8%) [DERIVED].",
              accent=INDIGO, bg_rgb=INDIGO_SOFT)
    _add_arrow(s, 8.68, 3.05, 0.45, 0.55)
    _add_card(s, 9.20, 1.55, 3.60, 3.70, "3. CO2 & PM2.5 Động",
              "CU-BEMS không gắn cảm biến CO2/PM2.5 -> tính qua phương trình cân bằng vật chất mỗi 5 phút trong Twin [SIMULATED].",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _add_rect(s, 0.50, 5.48, 12.3, 1.25, WHITE, border_rgb=BORDER, bw=1.2)
    _add_text(s, 0.70, 5.58, 11.9, 1.05,
              "Công thức minh bạch: ΔCO2 = (+18 L/h/người hô hấp − thông gió tươi Q_vent) trên phòng 140 m³; ΔPM2.5 theo lọc CADR HEPA.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Slide này giải thích tường minh: đúng là khi gom CU-BEMS về 1 giờ ta chỉ thu được nhiệt độ, độ ẩm và điện năng thực tế chứ file CSV không chứa CO2 hay bụi PM2.5. Do đó chúng ta suy luận số người từ tải điện rồi giải phương trình vật lý cân bằng CO2 (18 lít/giờ/người) và lọc bụi HEPA mỗi 5 phút bên trong Digital Twin.")

    # --- SLIDE 12: Q-LEARNING VS DOUBLE Q-LEARNING ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 12, "D1 • THUẬT TOÁN CỐT LÕI", "Cơ Chế Học Tăng Cường: Q-Learning Đơn vs Double Q-Learning")
    _add_picture_fit(s, fig_dir / "fig_09_q_vs_double_q_update.png", 0.50, 1.50, 7.6, 4.3)
    _add_card(s, 8.30, 1.50, 4.5, 2.08, "Single Q-Learning (QL)",
              "Chung 1 bảng Q vừa chọn max vừa định giá -> dễ thổi phồng nhiễu (Maximization Bias).",
              accent=ROSE)
    _add_card(s, 8.30, 3.70, 4.5, 2.15, "Double Q-Learning (DQL)",
              "Tách QA & QB cập nhật chéo 50% giúp định giá khách quan, ổn định qua 10 seeds.",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _add_rect(s, 0.50, 5.98, 12.3, 0.78, CYAN_SOFT, border_rgb=CYAN, bw=1.2)
    _add_text(s, 0.70, 6.02, 11.9, 0.70,
              f"Lịch khám phá Epsilon (Protocol v2): Từ 1.0 giảm dần về 0.02 qua {hp_dql['n_episodes']} tập (lr = {hp_dql['learning_rate']}).",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Ở Q-Learning đơn, cùng một bảng Q vừa dùng argmax chọn hành động vừa lấy chính nó làm mục tiêu nên dễ bị thổi phồng giá trị. Double Q-Learning tách làm hai bảng QA và QB cập nhật chéo 50/50 nhằm khử thiên lệch. Cả hai thuật toán dùng chung lịch epsilon từ 1.0 về 0.02.")

    # --- SLIDE 13 & 14 (Yêu cầu 9): KHÔNG GIAN TRẠNG THÁI 720 Ô — TÊN GỌI + NGƯỠNG RÕ RÀNG KHÔNG BỊ LÚ ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 13, "D2 • TRẠNG THÁI 720 Ô (1/2)", "3 Chiều Khí Hậu Trong Phòng: Đủ Tên Mức & Ngưỡng Đo")
    _add_table(
        s, 0.45, 1.50, 12.43, 3.85,
        ["Chiều quan sát", "Số mức", "Phân giỏ (Tên mức + Ngưỡng giá trị chính xác)"],
        [
            ["Nhiệt độ phòng (Tin)", "5", "0: Lạnh (<21°C) | 1: Mát [21–24) | 2: Chuẩn [24–26)\n3: Ấm [26–28.5) | 4: Nóng bức (>=28.5°C)"],
            ["Độ ẩm phòng (RH)", "3", "0: Khô (<40%)\n1: Thoải mái [40–65%) | 2: Ẩm ướt (>=65%)"],
            ["Nồng độ khí CO2", "3", "0: Trong lành (<800 ppm)\n1: Trung bình [800–1200 ppm) | 2: Ngạt (>=1200 ppm)"],
        ],
        [3.10, 1.25, 8.08], pt=20.0
    )
    _add_rect(s, 0.45, 5.58, 12.43, 1.05, CYAN_SOFT, border_rgb=CYAN, bw=1.2)
    _add_text(s, 0.65, 5.65, 12.03, 0.90,
              "Tổ hợp khí hậu phòng = 5 (Nhiệt) × 3 (Ẩm) × 3 (CO2) = 45 trạng thái vi khí hậu cốt lõi theo chuẩn ASHRAE.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Để tránh bị rối khi chia 720 giỏ, nhóm trình bày đầy đủ cả tên mức lẫn ngưỡng số chính xác trong code. Slide này là 3 chiều vi khí hậu bên trong phòng: Nhiệt độ chia 5 mức từ Lạnh <21°C đến Nóng >=28.5°C; Độ ẩm 3 mức Khô/Thoải mái/Ẩm ướt; CO2 3 mức Sạch/Trung bình/Ngạt.")

    s = prs.slides.add_slide(blank)
    _chrome(s, 14, "D2 • TRẠNG THÁI 720 Ô (2/2)", "3 Chiều Bụi, Tải & Khung Giờ (Ghi Rõ Theo Giờ Cụ Thể)")
    _add_table(
        s, 0.45, 1.50, 12.43, 3.85,
        ["Chiều quan sát", "Số mức", "Phân giỏ (Tên mức + Ngưỡng giờ / điện cụ thể)"],
        [
            ["Bụi mịn PM2.5", "2", "0: Sạch (<= 35 µg/m³) | 1: Ô nhiễm (> 35 µg/m³)"],
            ["Tải nội sinh / Người", "2", "0: Trống (< 0.50 kW) | 1: Có người (>= 0.50 kW, khớp 96.8%)"],
            ["Khung giờ & Biểu giá", "4", "0: Đêm/Cuối tuần (22h–06h & T7, CN) | 1: Sáng (06h–09h)\n2: Cao điểm 1.4x (09h–17h T2–T6) | 3: Tối (17h–22h T2–T6)"],
        ],
        [3.10, 1.25, 8.08], pt=20.0
    )
    _add_rect(s, 0.45, 5.58, 12.43, 1.05, EMERALD_SOFT, border_rgb=EMERALD, bw=1.2)
    _add_text(s, 0.65, 5.65, 12.03, 0.90,
              "Tổng tích Đề-các: 5 × 3 × 3 × 2 × 2 × 4 = 720 Ô. Ví dụ Ô 342 = [Nhiệt Chuẩn, Ẩm Chuẩn, CO2 Sạch, Bụi Sạch, Có người, 09h–17h].",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Slide 14 là 3 chiều còn lại kèm ví dụ giải mã cụ thể. Đặc biệt khung giờ được định nghĩa rất rõ ràng theo đồng hồ: mức 0 là đêm từ 22h đến 6h sáng hoặc cả ngày Thứ 7–Chủ Nhật; mức 1 là buổi sáng 6h–9h; mức 2 là giờ cao điểm giá điện 1.4 lần từ 9h–17h Thứ 2–Thứ 6; mức 3 là tối 17h–22h.")

    # --- SLIDES 15, 16, 17: 12 HÀNH ĐỘNG CHIA 3 SLIDE (A0-A3, A4-A7, A8-A11) ---
    def _short_dev_txt(aid: int) -> str:
        raw = ACTION_META_VI[aid]["devices_vi"]
        return raw.replace("Cửa: Đóng | ", "").replace("Cửa sổ: MỞ | ", "").replace("Sưởi: Tắt | ", "").replace(" | Ẩm: Tắt", "")

    action_slides = [
        (15, "D3 • KHÔNG GIAN HÀNH ĐỘNG (1/3)", "Thiết lập Thiết bị & Cửa sổ: A0 – A3 (Chờ & Làm mát)", range(0, 4),
         "Nhóm hành động làm mát cốt lõi: A1 (25°C chuẩn), A2 (23°C mạnh giờ nóng), A3 (27°C tiết kiệm giờ cao điểm)."),
        (16, "D4 • KHÔNG GIAN HÀNH ĐỘNG (2/3)", "Thiết lập Thiết bị & Cửa sổ: A4 – A7 (Cửa sổ, Xả CO2 & Sưởi)", range(4, 8),
         "Điểm sáng A4 (Mở cửa sổ đón gió trời) & A5 (Quạt 500 + HEPA 450 + AC 25°C để xả CO2 mà không bị nóng phòng!)."),
        (17, "D5 • KHÔNG GIAN HÀNH ĐỘNG (3/3)", "Thiết lập Thiết bị & Cửa sổ: A8 – A11 (Ẩm, Lọc HEPA & Đêm)", range(8, 12),
         "Đóng gói thành 12 mẫu giúp loại bỏ hàng nghìn tổ hợp rác xung đột (ví dụ vừa bật máy lạnh vừa bật máy sưởi)."),
    ]
    for s_idx, pill_s, title_s, id_rng, note_bar in action_slides:
        s = prs.slides.add_slide(blank)
        _chrome(s, s_idx, pill_s, title_s)
        sub_rows = [
            [f"A{aid}", ACTION_META_VI[aid]["name_only_vi"], ACTION_META_VI[aid]["window_vi"], _short_dev_txt(aid)]
            for aid in id_rng
        ]
        _add_table(
            s, 0.45, 1.50, 12.43, 3.65,
            ["Mã", "Tên Hành động (Tiếng Việt)", "Cửa sổ", "Cấu hình cụ thể trên 7 thiết bị"],
            sub_rows, [0.90, 4.10, 1.25, 6.18], pt=20.0
        )
        _add_rect(s, 0.45, 5.48, 12.43, 1.20, CYAN_SOFT, border_rgb=CYAN, bw=1.3)
        _add_text(s, 0.65, 5.56, 12.03, 1.02, note_bar, pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
        _notes(s, f"{title_s}: Mỗi mã lệnh A0..A11 điều khiển đồng bộ cả Cửa sổ lẫn công suất từng thiết bị. {note_bar}")

    # --- SLIDE 18: ĐIỂM THƯỞNG & PHẠT THEO ĐỘ QUAN TRỌNG ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 18, "E • HÀM ĐIỂM THƯỞNG & PHẠT", "Ý Nghĩa Phân Tầng Trọng Số 7 Thành Phần Trong Hàm Reward")
    _add_table(
        s, 0.45, 1.48, 12.43, 4.15,
        ["Ưu tiên", "Thành phần & Trọng số", "Thưởng / Phạt", "Vì sao chọn mức điểm này?"],
        [
            ["1 (Tối thượng)", "An toàn (w = 2.00)", "Phạt -10.0 + bậc 2", "Áp đảo mọi phần thưởng để cấm lệnh nguy hiểm"],
            ["2 (Nhiệt độ)", "Nhiệt độ (w = 1.00)", "+0.50 trong [24,26]°C", "Thước đo tiện nghi người dùng cảm nhận rõ nhất"],
            ["3 (Bền/Khí)", "Đảo nấc & CO2", "-0.35 | CO2 w=0.30", "Chống bật/tắt máy nén & kích thích xả CO2 (A5)"],
            ["4 (Phụ trợ)", "Độ ẩm & Bụi PM2.5", "Mỗi loại w=0.25", "Kiểm soát ẩm mốc, khô da và bụi mịn hô hấp"],
            ["5 (Chi phí)", "Điện năng (w = 0.12)", "-0.12 × kW × 1.4x", "Chỉ tiết kiệm khi an toàn & tiện nghi đã đạt"],
        ],
        [2.05, 3.05, 2.55, 4.78], pt=20.0
    )
    _add_rect(s, 0.45, 5.82, 12.43, 0.92, INDIGO_SOFT, border_rgb=INDIGO, bw=1.2)
    _add_text(s, 0.65, 5.88, 12.03, 0.80,
              "Giờ làm việc ×2.0 vs Đêm ×0.5: Khi phòng trống, AI tự chuyển sang chế độ tiết kiệm vì phạt nhiệt không còn nặng gấp đôi.",
              pt=20.0, bold=True, color=INDIGO, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Trọng số 7 thành phần tuân theo tháp ưu tiên vận hành an toàn: nặng nhất là An toàn (w=2.0) để triệt tiêu hành vi cực đoan; tiếp theo là Nhiệt độ w=1.0; chống đảo nấc 0.35 và CO2 0.30; cuối cùng mới là tiền điện 0.12 nhân biểu giá 1.4x.")

    # --- SLIDE 19: BẢNG Q-TABLE THỰC TẾ SAU TRAIN ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 19, "F • BẢNG Q-TABLE SAU TRAIN", "Trích Xuất Ma Trận Q-Table Thật Từ Checkpoint Seed 0")
    _add_picture_fit(s, fig_dir / "fig_19_qtable_comparison.png", 0.45, 1.45, 8.75, 5.35)
    _add_card(s, 9.35, 1.55, 3.55, 2.45, "Cách AI Đọc Bảng Q",
              "Gặp Ô 502 (Nóng+CO2), DQL tra hàng 502 -> chọn A2 (+6.6) & tránh A0 (-7.2).",
              accent=CYAN)
    _add_card(s, 9.35, 4.20, 3.55, 2.55, "DQL vs QL Thực tế",
              "Cùng chọn lệnh chuẩn ở giờ cao điểm, nhưng Q của QL cao hơn & phân tán hơn DQL.",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _notes(s, "Hình H19 trích trực tiếp giá trị Q-Table từ file model sau huấn luyện. Ở hàng Ô 342 (phòng mát giờ cao điểm), AI gán +9.8 cho A5; khi nóng lên Ô 502, giá trị cao nhất chuyển ngay sang A2 (+6.6), còn tắt máy A0 bị âm nặng (-7.2).")

    # --- SLIDE 20: TỪ 720 BINS -> TEST LIÊN TỤC ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 20, "G1 • CƠ CHẾ KIỂM THỬ THỰC", "720 Giỏ Dùng Để Tra Q-Table — Test Chạy Hoàn Toàn Số Thực!")
    _add_card(s, 0.50, 1.65, 3.5, 3.55, "1. Tập Test-A/B CSV",
              "Đọc nhiệt, ẩm & tải điện liên tục theo từng giờ của tòa nhà CU-BEMS 2018 & 2019.",
              accent=CYAN)
    _add_arrow(s, 4.10, 3.05, 0.65, 0.60)
    _add_card(s, 4.85, 1.65, 3.6, 3.55, "2. Tra Chỉ Mục Ô",
              "Số thực quy về chỉ mục 0..719 -> tra argmax Q(s,:) -> đóng/mở 7 thiết bị & cửa sổ.",
              accent=INDIGO)
    _add_arrow(s, 8.55, 3.05, 0.65, 0.60)
    _add_card(s, 9.30, 1.65, 3.5, 3.55, "3. Đo Vật Lý Thật",
              "Tính tích phân 5 phút ra kWh/ngày, % giờ vượt 26°C hoặc 1000 ppm để so Baseline.",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _add_rect(s, 0.50, 5.48, 12.3, 1.22, WHITE, border_rgb=BORDER, bw=1.2)
    _add_text(s, 0.70, 5.58, 11.9, 1.0,
              "Kết luận rõ ràng: Không hề có chuyện 'làm tròn rồi tự chấm điểm'; 720 ô chỉ là bản đồ chọn lệnh cho mô hình vật lý liên tục.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "720 giỏ chỉ đóng vai trò địa chỉ tra bảng Q-Table để Agent chọn hành động. Còn toàn bộ kết quả tiêu thụ điện kWh/ngày, % vi phạm nhiệt và CO2 được Digital Twin tính tích phân trên số thực liên tục hoàn toàn công bằng.")

    # --- SLIDE 21: VÍ DỤ QUỸ ĐẠO 24H THỰC TẾ ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 21, "G2 • VÍ DỤ VẬN HÀNH 24H", "Quỹ Đạo Thật Test-A + Phối Hợp 6 Luật Bảo Vệ Safety Gateway")
    _add_picture_fit(s, fig_dir / "fig_16_24h_trajectory_comparison.png", 0.45, 1.45, 8.5, 5.4)
    _add_card(s, 9.15, 1.55, 3.75, 2.45, "Diễn biến Thực tế",
              "09h–17h có người: DQL bật A5 giữ nhiệt [24,26]°C & CO2 phẳng ~690 ppm dưới mốc 1000.",
              accent=EMERALD)
    _add_card(s, 9.15, 4.20, 3.75, 2.55, "6 Khóa An toàn Cứng",
              "Chặn quá nhiệt >=29.5°C, ngạt >=1600 ppm, khóa cửa sổ đêm/mưa, chống đảo nấc nén.",
              accent=ROSE, bg_rgb=ROSE_SOFT)
    _notes(s, "Trên quỹ đạo 24 giờ thực từ dữ liệu Test-A (vùng vàng là giờ có người), đường xanh của DQL giữ nhiệt nằm gọn trong [24, 26]°C và giữ CO2 dưới 800 ppm, vượt xa Đặt cố định 25°C hay Rơ-le nhiệt.")

    # --- SLIDE 22: RANDOM SEED & 4 BỘ ĐỐI CHỨNG TIẾNG VIỆT ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 22, "H1 • SEED & ĐỐI CHỨNG", "Random Seed Là Gì? Giải Thích 4 Bộ Điều Khiển Cơ Sở")
    _add_card(s, 0.50, 1.55, 4.7, 5.20, "Random Seed (0..9) là gì?",
              "• Cố định chuỗi số ngẫu nhiên khi thám hiểm, cập nhật QA/QB và chọn 8 ngày kiểm thử.\n"
              "• Chạy 10 Seeds × 8 ngày = 80 ngày/thuật toán.\n"
              "• Tính Trung bình ± Độ lệch chuẩn (Mean ± SD) loại bỏ ăn may 1 lần chạy.",
              accent=INDIGO, bg_rgb=INDIGO_SOFT)
    _add_table(
        s, 5.45, 1.55, 7.35, 5.20,
        ["Tên Tiếng Việt", "Cách thức hoạt động"],
        [
            ["Đặt cố định 25°C", "Luôn giữ A1 (AC 25°C + gió nhẹ 150 m³/h) suốt 24/24h"],
            ["Bộ luật IF-THEN", "Tập ngưỡng cứng kỹ sư viết sẵn cho T, RH, CO2, giờ"],
            ["Rơ-le nhiệt truyền thống", "Chỉ bật/tắt AC theo nhiệt độ [21,26]°C, không điều tiết gió"],
            ["Chính sách ngẫu nhiên", "Chọn bừa 1/12 hành động để chứng minh AI có học"],
        ],
        [3.15, 4.20], pt=20.0
    )
    _notes(s, "Mỗi biểu đồ so sánh tổng hợp từ 10 Random Seed độc lập (80 ngày mô phỏng mỗi thuật toán) thay vì ăn may 1 lần chạy. Các baseline được Việt hóa rõ gồm Đặt cố định 25°C, Bộ luật IF-THEN, Rơ-le nhiệt truyền thống và Chính sách ngẫu nhiên.")

    # --- SLIDE 23: HỘI TỤ & KIỂM CHỨNG MAXBIAS ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 23, "H2 • HUẤN LUYỆN & MAXBIAS", "Quá Trình Hội Tụ 10 Seeds & Kiểm Chứng Khử Thiên Lệch")
    _add_picture_fit(s, fig_dir / "fig_11_training_convergence_curves.png", 0.45, 1.48, 6.2, 4.15)
    _add_picture_fit(s, fig_dir / "fig_15_maximization_bias_verification.png", 6.75, 1.48, 6.1, 4.15)
    _add_rect(s, 0.45, 5.78, 12.4, 0.95, CYAN_SOFT, border_rgb=CYAN, bw=1.2)
    _add_text(s, 0.65, 5.85, 12.0, 0.80,
              f"Kết quả kiểm chứng: Trên Toy MDP, DQL giảm thiên lệch chọn nhầm từ {be['ql_mean_bias']:.2f}% (QL) xuống {be['dql_mean_bias']:.2f}%; trên HVAC cả 2 hội tụ mượt về ε=0.02.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Đường cong huấn luyện 3000 episodes qua 10 seeds hội tụ mượt về epsilon=0.02 với điểm thưởng lên vùng +15 đến +16. Bên phải, bài kiểm tra Toy MDP chứng minh DQL giảm chọn nhầm nhánh từ 38.09% xuống còn 12.32%.")

    # --- SLIDE 24: ĐỐI ĐẦU TOÀN DIỆN TEST-A ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 24, "H3 • KẾT QUẢ TEST-A", "So Sánh Thực Tế 6 Bộ Điều Khiển (10 Seeds × 8 Ngày = 80 Ngày)")
    _add_picture_fit(s, fig_dir / "fig_12_main_benchmark_metrics.png", 0.35, 1.50, 5.40, 5.20)
    _add_table(
        s, 5.88, 1.50, 7.12, 5.15,
        ["Bộ điều khiển", "Reward", "kWh", "Lỗi T°C", "Lỗi CO2"],
        [
            ["Double Q-Learning", dql_a["mean_reward"], "20.00", "2.0%", "1.0%"],
            ["Single Q-Learning", ql_a["mean_reward"], "20.73", "3.1%", "2.5%"],
            ["Đặt cố định 25°C", fix_a["mean_reward"], "13.21", "1.1%", "74.4%"],
            ["Bộ luật IF-THEN", rb_a["mean_reward"], "15.26", "3.3%", "61.7%"],
            ["Rơ-le nhiệt", thm_a["mean_reward"], "11.57", "48.4%", "63.1%"],
            ["Ngẫu nhiên", rnd_a["mean_reward"], "16.42", "67.3%", "51.8%"],
        ],
        [2.32, 1.45, 1.05, 1.15, 1.15], pt=20.0
    )
    _notes(s, "Trên tập Test-A độc lập, hai mô hình Học tăng cường dẫn đầu tuyệt đối: DQL đạt 16.02 ± 1.03 điểm (20.00 kWh/ngày, lỗi nhiệt 2.0%, lỗi CO2 1.0%) và QL đạt 15.42 ± 2.11 điểm (20.73 kWh/ngày). Đặt cố định 25°C (13.21 kWh/ngày) để CO2 ngạt tới 74.4% thời gian.")

    # --- SLIDE 25: PHÂN PHỐI & THỐNG KÊ CÔNG BẰNG ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 25, "H4 • CÔNG BẰNG THỐNG KÊ", "Phân Phối Hành Động & Kiểm Định Cặp DQL vs QL (Branch 2)")
    _add_picture_fit(s, fig_dir / "fig_18_action_distribution.png", 0.45, 1.45, 6.5, 4.05)
    _add_picture_fit(s, fig_dir / "fig_14_paired_dql_vs_ql_stats.png", 7.05, 1.45, 5.85, 4.05)
    _add_rect(s, 0.45, 5.65, 12.4, 1.15, EMERALD_SOFT, border_rgb=EMERALD, bw=1.4)
    _add_text(s, 0.65, 5.72, 12.0, 1.00,
              f"Kết luận (Branch {branch}): DQL ({dql_a['mean_reward']}) & QL ({ql_a['mean_reward']}) tương đương trung bình (p={st['paired_t']['p_value']:.4f}), nhưng DQL ổn định hơn (~4× ít phương sai) & thắng xa mọi Baseline (p<1e-7).",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Kiểm định ghép cặp 10 seeds cho p=0.4274 (Nhánh 2 — tương đương kỳ vọng trung bình trên bảng rời rạc 720 ô), nhưng DQL ổn định phương sai hơn ~4 lần so với QL và cả 2 vượt trội tuyệt đối mọi baseline với p < 10^-7.")

    # --- SLIDE 26: CHUYỂN GIAO MIỀN TEST-B & TEST-C ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 26, "H5 • CHUYỂN GIAO MIỀN", "Đánh Giá Chuyển Giao Miền 2019 (Test-B) & Test-C Cực Đoan")
    _add_picture_fit(s, fig_dir / "fig_17_domain_transfer_and_robustness.png", 0.45, 1.48, 12.4, 4.20)
    _add_rect(s, 0.45, 5.82, 12.4, 0.96, CYAN_SOFT, border_rgb=CYAN, bw=1.2)
    _add_text(s, 0.65, 5.88, 12.0, 0.84,
              f"Test-B trọn năm 2019 (6,964 giờ, không học lại): DQL dẫn đầu Reward {dql_b['mean_reward']} ({dql_b['energy_kwh_per_day']} kWh/ngày) và giữ vững an toàn ở các bài Stress Nắng nóng / Đông người / Bụi mịn.",
              pt=20.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Mang chính sách đã học năm 2018 áp dụng trực tiếp cho toàn năm 2019 (6,964 giờ Test-B) và 3 kịch bản khắc nghiệt Test-C mà không học lại, DQL vẫn dẫn đầu và vận hành an toàn dưới lớp bảo vệ Safety Gateway.")

    # --- SLIDE 27: CẢM HỨNG GIAO DIỆN CONTROL ROOM ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 27, "I • TRỰC QUAN HÓA CONTROL ROOM", "Ý Tưởng & Cảm Hứng Thiết Kế Giao Diện Giám Sát Tiếng Việt")
    _add_picture_fit(s, assets_dir / "gui_raw_vi.png", 0.45, 1.48, 8.15, 5.3)
    _add_card(s, 8.78, 1.48, 4.10, 1.72, "Cảm hứng SCADA / BMS",
              "Nền tối chống mỏi mắt kỹ sư; 8 thẻ KPI đổi màu theo ngưỡng cảnh báo.",
              accent=CYAN)
    _add_card(s, 8.78, 3.30, 4.10, 1.74, "Việt hóa & Mở rộng Bước",
              "Việt hóa nhãn & mở rộng Bảng Bước kèm cột Thiết lập Cửa sổ/Thiết bị.",
              accent=EMERALD)
    _add_card(s, 8.78, 5.14, 4.10, 1.72, "Giải mã Hộp đen AI",
              "Nhấp 1 dòng trong bảng để soi trạng thái s, giá trị QA/QB & điểm thưởng.",
              accent=INDIGO)
    _notes(s, "Giao diện Digital Twin lấy cảm hứng từ hệ thống SCADA/BMS công nghiệp, được Việt hóa toàn bộ, mở rộng Bảng Lịch sử Bước để dễ theo dõi trực tiếp và hiển thị cụ thể lệnh cài đặt trên từng thiết bị lẫn cửa sổ.")

    # --- SLIDE 28: HƯỚNG DẪN CHẠY DEMO & ĐỌC 6 KHU VỰC GIAO DIỆN ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 28, "J • HƯỚNG DẪN CHẠY DEMO", "Cách Chạy & Đọc 6 Khu Vực Trên Giao Diện Tiếng Việt")
    _add_picture_fit(s, assets_dir / "gui_demo_annotated_vi.png", 0.40, 1.42, 12.53, 4.65)
    _add_rect(s, 0.45, 6.14, 12.43, 0.72, DARK_BG, border_rgb=CYAN, bw=1.0)
    _add_text(s, 0.60, 6.18, 12.13, 0.64,
              "Lệnh: python run_live_simulation.py | Demo: Nhấn 'CHẠY 1 BƯỚC' (Khu 2) -> Xem Thiết bị (Khu 3), Q-Table (Khu 4) & Bảng Bước (Khu 6)",
              pt=20.0, bold=True, color=CYAN_BRIGHT, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Khi demo trước Hội đồng bằng lệnh python run_live_simulation.py: chọn phòng CU-BEMS 2018 ở Khu 2, nhấn 'CHẠY 1 BƯỚC' để dừng lại chỉ rõ trạng thái 7 thiết bị ở Khu 3, dòng tra Q-Table & thiết lập cửa sổ ở Khu 4, đồ thị ở Khu 5 và bảng lịch sử chi tiết ở Khu 6.")

    # --- SLIDE 29: KẾT QUẢ CUỐI CÙNG ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 29, "K • KẾT QUẢ CUỐI CÙNG", "Các Chỉ Số Vàng Khẳng Định Hiệu Quả Toàn Hệ Thống")
    _add_card(s, 0.50, 1.60, 2.9, 3.45, "16.02 ± 1.03",
              "Điểm thưởng Test-A cao nhất (vượt Cố định 25°C là 9.12 và Rơ-le -63.07).", accent=CYAN, bg_rgb=CYAN_SOFT, title_pt=25.0)
    _add_card(s, 3.63, 1.60, 2.9, 3.45, "1.0% vs 74.4%",
              "Kéo tỷ lệ ngạt khí CO2 >1000 ppm từ 74.4% (Cố định 25°C) xuống chỉ 1.0%.", accent=EMERALD, bg_rgb=EMERALD_SOFT, title_pt=25.0)
    _add_card(s, 6.76, 1.60, 2.9, 3.45, "0% Can thiệp",
              "Safety Gateway không phải ép sửa lỗi lần nào trên tập Test nhờ AI tự học an toàn.", accent=INDIGO, bg_rgb=INDIGO_SOFT, title_pt=25.0)
    _add_card(s, 9.90, 1.60, 2.9, 3.45, "100% Đạt Gate",
              "Đạt toàn bộ 8 cổng kiểm soát tự động & toàn bộ bộ kiểm thử đơn vị Pytest.", accent=AMBER, bg_rgb=AMBER_SOFT, title_pt=25.0)
    _add_rect(s, 0.50, 5.30, 12.3, 1.30, WHITE, border_rgb=BORDER, bw=1.2)
    _add_text(s, 0.70, 5.40, 11.9, 1.10,
              "Đồng bộ 100%: Mã nguồn Python ↔ Giao diện Tiếng Việt ↔ Báo cáo Word (Insert Table of Figures) ↔ Slide thuyết trình.",
              pt=21.0, bold=True, color=INK, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Tóm lại 4 kết quả nổi bật: điểm thưởng 16.02 đứng đầu, giảm ngạt CO2 từ 74.4% còn 1.0%, 0% vi phạm an toàn trên Test-A và đạt 100% kiểm thử tự động cùng 8 Gate kiểm định.")

    # --- SLIDE 30: ĐÃ LÀM ĐƯỢC & NHƯỢC ĐIỂM ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 30, "L & M • TỔNG KẾT & NHƯỢC ĐIỂM", "Đánh Giá Khách Quan: Thành Tựu Đạt Được vs Hạn Chế Hiện Tại")
    _add_card(s, 0.50, 1.55, 6.0, 5.10, "Đã Làm Được (Thành tựu)",
              "• Hiện thực hóa đầy đủ Digital Twin 7 thiết bị & Safety 6 luật.\n"
              "• Huấn luyện & kiểm định 10 Random Seeds trên CU-BEMS 2018-2019.\n"
              "• Việt hóa trọn vẹn GUI mở rộng, bảng 12 hành động & Q-Table.",
              accent=EMERALD, bg_rgb=EMERALD_SOFT)
    _add_card(s, 6.80, 1.55, 6.0, 5.10, "Nhược Điểm Còn Tồn Tại (Trung thực)",
              "• Chưa đấu nối thử nghiệm trên phòng thật hoặc phần cứng BMS/Arduino ngoài đời.\n"
              "• CO2 và PM2.5 tính qua phương trình mô phỏng (do CU-BEMS thiếu cảm biến).\n"
              "• Bảng Q rời rạc 720 ô hạn chế độ mịn khi mở rộng nhiều phòng.",
              accent=ROSE, bg_rgb=ROSE_SOFT)
    _notes(s, "Nhóm đã hoàn thiện toàn bộ quy trình từ dữ liệu thực, thuật toán DQL/QL, Safety Gateway đến giao diện tiếng Việt. Hạn chế trung thực là chưa chạy trên phòng thật ngoài đời, CO2/PM2.5 là phương trình mô phỏng và Q-Table mới ở dạng rời rạc 720 ô.")

    # --- SLIDE 31: HƯỚNG PHÁT TRIỂN ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 31, "N • HƯỚNG PHÁT TRIỂN", "Lộ Trình Đưa Bộ Não AI Xuống Phần Cứng Thực Tế")
    _add_card(s, 0.50, 1.60, 2.8, 3.70, "01 • Phần Cứng",
              "Đưa lệnh A0–A11 từ Python xuống Arduino / ESP32 điều khiển Servo 90° & Relay thực tế.",
              accent=CYAN)
    _add_arrow(s, 3.35, 3.15, 0.35, 0.50)
    _add_card(s, 3.75, 1.60, 2.8, 3.70, "02 • Cảm Biến",
              "Lắp cảm biến CO2 (MH-Z19), bụi PM2.5 và nhiệt độ thật thay cho biến giả lập.",
              accent=EMERALD)
    _add_arrow(s, 6.60, 3.15, 0.35, 0.50)
    _add_card(s, 7.00, 1.60, 2.8, 3.70, "03 • Deep RL",
              "Dùng DQN / PPO nhận thẳng trạng thái liên tục thay cho chia 720 giỏ rời rạc.",
              accent=INDIGO)
    _add_arrow(s, 9.85, 3.15, 0.35, 0.50)
    _add_card(s, 10.25, 1.60, 2.6, 3.70, "04 • Đa Phòng",
              "Mở rộng phối hợp nhiều phòng và tầng cùng hệ thống BMS tòa nhà.",
              accent=AMBER)
    _add_rect(s, 0.50, 5.55, 12.3, 1.15, WHITE, border_rgb=CYAN, bw=1.3)
    _add_text(s, 0.70, 5.65, 11.9, 0.95,
              "Khép kín vòng lặp khát vọng ban đầu: Từ SmartHouse Arduino thô sơ -> Bộ não AI vững chắc -> Hệ thống nhà thông minh thực thụ.",
              pt=20.0, bold=True, color=CYAN, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Bước phát triển tiếp theo là kết nối ngược lệnh từ Python xuống Arduino/ESP32 điều khiển Servo 90 độ và relay điều hòa thật, lắp cảm biến CO2/PM2.5 thật, đồng thời nâng cấp lên Deep Q-Network để điều khiển đa phòng liên tục.")

    # --- SLIDE 32: KẾT & CẢM ƠN ---
    s = prs.slides.add_slide(blank)
    _chrome(s, 32, "", "", dark=True)
    _add_plain_rect(s, 0.0, 0.0, 0.25, 7.5, EMERALD)
    _add_text(s, 0.85, 1.10, 11.6, 1.1, "XIN TRÂN TRỌNG CẢM ƠN!", pt=42.0, bold=True, color=CYAN_BRIGHT, align=PP_ALIGN.CENTER)
    _add_text(s, 0.85, 2.30, 11.6, 0.90,
              f"Kính mời Thầy {sup_full} & Hội đồng đặt câu hỏi và xem Demo trực tiếp",
              pt=24.0, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _add_rect(s, 1.45, 3.40, 10.4, 3.05, RGBColor(30, 41, 59), border_rgb=CYAN, bw=1.5)
    _add_text(s, 1.75, 3.55, 9.8, 2.75,
              "• Tái lập thực nghiệm 100%: python run_all.py  |  Demo GUI: python run_live_simulation.py\n"
              "• Nhóm 2: Nguyễn Vũ Đức Anh • Trần Thị Thanh Ngân • Võ Thanh Trường • Võ Thị Diễm My\n"
              "• Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) — Khoa Công nghệ Thông tin",
              pt=20.0, color=CYAN_SOFT, line_spacing=1.22, valign=MSO_ANCHOR.MIDDLE)
    _notes(s, "Nhóm 2 xin chân thành cảm ơn Thầy TS. Nguyễn Thanh Long đã tận tình hướng dẫn và cảm ơn Hội đồng đã lắng nghe. Sau đây nhóm xin phép chạy demo giao diện tiếng Việt thực tế và trả lời các câu hỏi của Hội đồng.")

    # Hard self-verification: guarantee NO run anywhere is < 20.0pt
    violations = []
    for si, slide in enumerate(prs.slides, 1):
        for shp in slide.shapes:
            if shp.has_text_frame:
                for p in shp.text_frame.paragraphs:
                    for r in p.runs:
                        if r.text.strip() and r.font.size and round(r.font.size.pt, 1) < 20.0:
                            violations.append((si, r.font.size.pt, r.text[:25]))
            if shp.has_table:
                for row in shp.table.rows:
                    for cell in row.cells:
                        for p in cell.text_frame.paragraphs:
                            for r in p.runs:
                                if r.text.strip() and r.font.size and round(r.font.size.pt, 1) < 20.0:
                                    violations.append((si, r.font.size.pt, r.text[:25]))
    if violations:
        raise RuntimeError(f"Font size violation < 20pt detected: {violations[:5]}")

    out_path = REPO_ROOT / "doc" / "BAO_CAO_THUYET_TRINH_FINAL.pptx"
    prs.save(out_path)
    print(f"[*] Generated updated 32-slide >=20pt PowerPoint deck -> {out_path} ({len(prs.slides)} slides, 0 font violations)")


if __name__ == "__main__":
    main()
