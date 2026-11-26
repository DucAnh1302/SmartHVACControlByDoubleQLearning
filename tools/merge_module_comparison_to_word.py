"""Merge comparative module evaluation & percentage methodology cleanly into doc/BAO_CAO_DO_AN_FINAL.docx.

Inserts Section 6.8 at the end of Chapter 6 (right before CHƯƠNG 7), preserving:
- Native Word Heading 2 / Heading 3 hierarchy and static MỤC LỤC page sync
- Genuine Word SEQ Bảng dynamic captions + Table of Tables auto-update
- Native Word OMML equations (<m:oMath>, <m:oMathPara>)
- Exact A4 printable width (9072 dxa = 6.30 in) rescaling, cantSplit rows & repeating headers
"""

from copy import deepcopy
from pathlib import Path
import sys
import docx
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "doc" / "BAO_CAO_DO_AN_FINAL.docx"
SOURCE_PATH = REPO_ROOT / "doc" / "TONG_HOP_DANH_GIA_SO_SANH_MODULE_HVAC.docx"
TARGET_WIDTH_DXA = 9072  # 8.27" - (1.18" + 0.79") = 6.30" printable width


def _rescale_table(tbl_el, target_w_dxa: int = TARGET_WIDTH_DXA, repeat_header: bool = False, compact_pad: bool = False):
    """Proportionally rescale tblW, tblGrid, and tcW to fit printable width and prevent split/overflow."""
    grid_cols = tbl_el.xpath("./w:tblGrid/w:gridCol")
    old_widths = [int(gc.get(qn("w:w")) or 1000) for gc in grid_cols]
    old_total = max(1, sum(old_widths))
    scale = target_w_dxa / old_total

    new_widths = [max(320, int(round(w * scale))) for w in old_widths]
    if compact_pad and len(new_widths) == 10:
        # Give the trailing 'Override (%) ↓' column enough width so 'Override' stays on one line
        if new_widths[9] < 900:
            delta = 900 - new_widths[9]
            new_widths[9] += delta
            if new_widths[1] > delta + 1000:
                new_widths[1] -= delta
            else:
                new_widths[2] -= delta
    diff = target_w_dxa - sum(new_widths)
    if new_widths:
        idx_max = max(range(len(new_widths)), key=lambda i: new_widths[i])
        new_widths[idx_max] += diff

    for gc, nw in zip(grid_cols, new_widths):
        gc.set(qn("w:w"), str(nw))

    tbl_pr = tbl_el.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        tbl_el.insert(0, tbl_pr)
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(target_w_dxa))
    tbl_w.set(qn("w:type"), "dxa")

    jc = tbl_pr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        tbl_pr.append(jc)
    jc.set(qn("w:val"), "center")

    if compact_pad:
        tbl_mar = tbl_pr.find(qn("w:tblCellMar"))
        if tbl_mar is None:
            tbl_mar = OxmlElement("w:tblCellMar")
            tbl_pr.append(tbl_mar)
        for side, val in [("top", "55"), ("bottom", "55"), ("left", "45"), ("right", "45")]:
            el = tbl_mar.find(qn(f"w:{side}"))
            if el is None:
                el = OxmlElement(f"w:{side}")
                tbl_mar.append(el)
            el.set(qn("w:w"), val)
            el.set(qn("w:type"), "dxa")

    rows = tbl_el.xpath("./w:tr")
    for r_idx, tr in enumerate(rows):
        tr_pr = tr.find(qn("w:trPr"))
        if tr_pr is None:
            tr_pr = OxmlElement("w:trPr")
            tr.insert(0, tr_pr)
        if tr_pr.find(qn("w:cantSplit")) is None:
            tr_pr.append(OxmlElement("w:cantSplit"))
        if repeat_header and r_idx == 0 and tr_pr.find(qn("w:tblHeader")) is None:
            tr_pr.append(OxmlElement("w:tblHeader"))

        col_cursor = 0
        for tc in tr.xpath("./w:tc"):
            tc_pr = tc.find(qn("w:tcPr"))
            span_el = tc_pr.find(qn("w:gridSpan")) if tc_pr is not None else None
            span = int(span_el.get(qn("w:val"))) if span_el is not None else 1
            cell_w = sum(new_widths[col_cursor : col_cursor + span])
            col_cursor += span
            if tc_pr is None:
                tc_pr = OxmlElement("w:tcPr")
                tc.insert(0, tc_pr)
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(cell_w))
            tc_w.set(qn("w:type"), "dxa")

            if compact_pad:
                for p in tc.xpath(".//w:p"):
                    for r in p.xpath("./w:r"):
                        r_pr = r.find(qn("w:rPr"))
                        if r_pr is None:
                            r_pr = OxmlElement("w:rPr")
                            r.insert(0, r_pr)
                        sz = r_pr.find(qn("w:sz"))
                        if sz is None:
                            sz = OxmlElement("w:sz")
                            r_pr.append(sz)
                        sz.set(qn("w:val"), "17")  # 8.5pt clean fit for 10-col comparison tables
                        sz_cs = r_pr.find(qn("w:szCs"))
                        if sz_cs is None:
                            sz_cs = OxmlElement("w:szCs")
                            r_pr.append(sz_cs)
                        sz_cs.set(qn("w:val"), "17")


def _make_seq_caption(doc: docx.Document, description: str) -> OxmlElement:
    """Create a genuine Word Caption paragraph with dynamic SEQ Bảng field."""
    p = doc.add_paragraph("")
    p.style = doc.styles["Caption"]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)

    r1 = p.add_run("Bảng ")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(10.5)
    r1.bold = True
    r1.italic = True

    fld = parse_xml(
        f'<w:fldSimple {nsdecls("w")} w:instr=" SEQ Bảng \\* ARABIC ">'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
        f'<w:b/><w:i/><w:sz w:val="21"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'
    )
    p._p.append(fld)

    r2 = p.add_run(f": {description}")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(10.5)
    r2.italic = True
    el = p._p
    el.getparent().remove(el)
    return el


def _make_para(doc: docx.Document, text: str, style_name: str = "Normal", bold: bool = False, keep_next: bool = False, space_before: int = 4, space_after: int = 4) -> OxmlElement:
    p = doc.add_paragraph(text)
    p.style = doc.styles[style_name]
    p.paragraph_format.keep_with_next = keep_next
    if style_name == "Normal":
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
    for r in p.runs:
        r.font.name = "Times New Roman"
        if style_name == "Normal":
            r.font.size = Pt(12)
        if bold:
            r.bold = True
    el = p._p
    el.getparent().remove(el)
    return el


def _normalize_copied_para(p_el) -> None:
    """Keep native OMML equations untouched while aligning body paragraphs cleanly to 12pt Times New Roman."""
    p_pr = p_el.find(qn("w:pPr"))
    if p_pr is None:
        p_pr = OxmlElement("w:pPr")
        p_el.insert(0, p_pr)
    for r in p_el.xpath("./w:r"):
        r_pr = r.find(qn("w:rPr"))
        if r_pr is None:
            r_pr = OxmlElement("w:rPr")
            r.insert(0, r_pr)
        rf = r_pr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            r_pr.insert(0, rf)
        rf.set(qn("w:ascii"), "Times New Roman")
        rf.set(qn("w:hAnsi"), "Times New Roman")
        sz = r_pr.find(qn("w:sz"))
        if sz is None:
            sz = OxmlElement("w:sz")
            r_pr.append(sz)
        sz.set(qn("w:val"), "24")  # 12pt consistent with Chapter 6


def merge() -> None:
    d1 = docx.Document(str(REPORT_PATH))
    d2 = docx.Document(str(SOURCE_PATH))

    # Idempotency: remove existing 6.8 block if previously merged
    anchor_idx = None
    start_68_idx = None
    for i, p in enumerate(d1.paragraphs):
        if p.text.strip().startswith("6.8.") and p.style.name == "Heading 2":
            start_68_idx = i
        if "CHƯƠNG 7:" in p.text.upper() and p.style.name == "Heading 1":
            anchor_idx = i
            break
    if anchor_idx is None:
        raise RuntimeError("Could not find CHƯƠNG 7 Heading 1 anchor inside BAO_CAO_DO_AN_FINAL.docx!")

    if start_68_idx is not None:
        start_el = d1.paragraphs[start_68_idx]._p
        end_el = d1.paragraphs[anchor_idx]._p
        body = d1.element.body
        idx_s = list(body).index(start_el)
        idx_e = list(body).index(end_el)
        for k in range(idx_e - 1, idx_s - 1, -1):
            body.remove(body[k])
        # Re-resolve anchor after deletion
        anchor_idx = next(i for i, p in enumerate(d1.paragraphs) if "CHƯƠNG 7:" in p.text.upper() and p.style.name == "Heading 1")

    anchor_el = d1.paragraphs[anchor_idx]._p
    insert_nodes: list = []

    # Section 6.8 Heading 2 + transition intro
    insert_nodes.append(_make_para(d1, "6.8. Tổng hợp So sánh Chi tiết 6 Mô hình và Phương pháp Tính Chỉ số Đánh giá", style_name="Heading 2", keep_next=True))
    insert_nodes.append(_make_para(
        d1,
        "Mục này tổng hợp toàn bộ kết quả định lượng từ các kịch bản Test-A (2018), Test-B (2019) và Test-C (Stress cực đoan) của 6 mô hình điều khiển, đồng thời chuẩn hóa hai tầng công thức tính phần trăm nhằm bảo đảm tính minh bạch tuyệt đối của mọi con số báo cáo trước Hội đồng.",
        style_name="Normal",
        space_before=4,
        space_after=6,
    ))

    # Map elements B[01..51] from TONG_HOP into cohesive Chapter 6 subsections
    subheadings_h3 = {
        1: "6.8.1. Kết luận tổng quan: Mô hình tối ưu theo từng kịch bản kiểm thử",
        7: "6.8.2. Bảng đối chiếu chi tiết 6 mô hình trên Test-A, Test-B, Test-C và Chênh lệch phần trăm",
        18: "6.8.3. Phương pháp luận Tầng 1: Cách tính các chỉ số phần trăm gốc (kèm ví dụ thế số)",
        36: "6.8.4. Phương pháp luận Tầng 2: Công thức tính phần trăm khi so sánh các mô hình với nhau",
    }
    sublabels_bold = {
        8: "a) Bảng so sánh chi tiết 6 mô hình trên dữ liệu thực tế chuẩn Test-A (2018) và Test-B (2019)",
        11: "b) Bảng so sánh chi tiết 6 mô hình trên 3 kịch bản Stress cực đoan Test-C",
        14: "c) Bảng chênh lệch phần trăm (%) và điểm phần trăm (pp) của Double Q-Learning so với 5 mô hình còn lại trên Test-A",
        20: "a) Công thức 1 — Tỷ lệ vi phạm Nhiệt độ, CO₂, PM₂.₅ và Độ ẩm RH (%)",
        26: "b) Công thức 2 — Tỷ lệ can thiệp Safety Gateway (%)",
        31: "c) Công thức 3 — Trung bình theo Seed và Trung bình ± Độ lệch chuẩn trên 10 Seeds",
        37: "a) Công thức 4 — So sánh đại lượng vật lý không có đơn vị % (Điện năng, Switching)",
        42: "b) Công thức 5 — So sánh các chỉ số vốn đã mang đơn vị % (Điểm phần trăm pp vs % Giảm tương đối)",
        47: "c) Công thức 6 — So sánh Mean Reward, Cohen's d_z và Tỷ lệ giảm phương sai",
    }
    table_captions = {
        4: "Tổng kết mô hình tối ưu nhất theo từng kịch bản kiểm thử (Protocol v2 — 10 Random Seeds)",
        9: "So sánh chi tiết 6 mô hình điều khiển trên dữ liệu thực tế Test-A (2018) và Test-B (2019)",
        12: "Đối chiếu 6 mô hình điều khiển trên 3 kịch bản Stress cực đoan Test-C",
        16: "Chênh lệch phần trăm (%) và điểm phần trăm (pp) của Double Q-Learning so với 5 mô hình còn lại trên Test-A",
    }

    body2 = list(d2.element.body)
    for b_idx, el in enumerate(body2):
        if b_idx == 0:
            continue  # Skip standalone banner of TONG_HOP
        if b_idx in subheadings_h3:
            insert_nodes.append(_make_para(d1, subheadings_h3[b_idx], style_name="Heading 3", keep_next=True))
            continue
        if b_idx in sublabels_bold:
            insert_nodes.append(_make_para(d1, sublabels_bold[b_idx], style_name="Normal", bold=True, keep_next=True, space_before=6, space_after=3))
            continue

        if isinstance(el, CT_P):
            p = docx.text.paragraph.Paragraph(el, d2)
            # Drop redundant blank spacer paragraphs before tables/subsections
            if not p.text.strip() and "oMath" not in el.xml:
                continue
            p_copy = deepcopy(el)
            _normalize_copied_para(p_copy)
            insert_nodes.append(p_copy)
        elif isinstance(el, CT_Tbl):
            if b_idx in table_captions:
                insert_nodes.append(_make_seq_caption(d1, table_captions[b_idx]))
            tbl_copy = deepcopy(el)
            is_wide_10col = b_idx in (9, 12)
            is_data_tbl = b_idx in table_captions
            _rescale_table(tbl_copy, target_w_dxa=TARGET_WIDTH_DXA, repeat_header=is_data_tbl, compact_pad=is_wide_10col)
            insert_nodes.append(tbl_copy)
            sp = _make_para(d1, "", style_name="Normal", space_before=0, space_after=2)
            insert_nodes.append(sp)

    for node in insert_nodes:
        anchor_el.addprevious(node)

    # Cleanly remove duplicate static plain-text TOC paragraphs sitting directly after <w:sdt> auto-TOC
    body = d1.element.body
    children = list(body)
    sdt_pos = next((i for i, el in enumerate(children) if el.tag.endswith("}sdt") and "TOC" in "".join(el.itertext())), None)
    bpc_pos = next(
        (i for i, el in enumerate(children)
         if sdt_pos is not None and i > sdt_pos and el.tag.endswith("}p")
         and "TRANG " not in "".join(el.itertext()).upper()
         and docx.text.paragraph.Paragraph(el, d1).text.strip().upper() == "BẢNG PHÂN CÔNG NHIỆM VỤ"),
        None,
    )
    if sdt_pos is not None and bpc_pos is not None and bpc_pos > sdt_pos + 1:
        to_remove = []
        for k in range(sdt_pos + 1, bpc_pos):
            txt = docx.text.paragraph.Paragraph(children[k], d1).text.strip()
            if not txt or "Trang " in txt:
                to_remove.append(children[k])
        for el in to_remove:
            body.remove(el)

    d1.save(str(REPORT_PATH))
    print("[*] Inserted Section 6.8 & cleaned duplicate TOC inside BAO_CAO_DO_AN_FINAL.docx")

    # Update genuine Word SDT TOC, SEQ fields, and List of Figures/Tables via Word COM
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc_com = word.Documents.Open(str(REPORT_PATH))
        doc_com.Repaginate()
        doc_com.Fields.Update()
        for tof in doc_com.TablesOfFigures:
            tof.Update()
        for toc in doc_com.TablesOfContents:
            toc.Update()
        doc_com.Save()
        doc_com.ExportAsFixedFormat(
            r"C:\Users\LENOVO\.gemini\antigravity\brain\bf8ad85e-f99b-4045-bd65-bc6346fa0b38\scratch\report_merged_v2.pdf",
            17,
        )
        try:
            doc_com.Close(False)
        except Exception:
            pass
        print("[*] Updated all Word auto-TOC, SEQ fields, List of Figures/Tables & exported PDF via Word COM")
    finally:
        try:
            word.Quit()
        except Exception:
            pass


if __name__ == "__main__":
    merge()
