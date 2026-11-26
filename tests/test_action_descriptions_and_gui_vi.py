"""Tests for Vietnamese action/controller mappings, expanded Vietnamese GUI, 20pt+ PPTX, and Word Captions/TOF."""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.action_descriptions_vi import (
    ACTION_META_VI,
    CONTROLLER_META_VI,
    format_action_devices_vi,
    format_action_short_vi,
)
from src.environment.multi_device_env import MULTI_DEVICE_ACTION_TEMPLATES


def test_action_meta_vi_covers_all_12_templates_accurately():
    assert len(ACTION_META_VI) == 12
    for aid, tmpl in MULTI_DEVICE_ACTION_TEMPLATES.items():
        meta = ACTION_META_VI[aid]
        assert meta["code"] == tmpl.name
        assert meta["short_vi"].startswith(f"A{aid}:")
        assert len(meta["name_only_vi"]) >= 6
        expected_win = "MỞ" if tmpl.window_open == 1 else "Đóng"
        assert meta["window_vi"] == expected_win
        assert format_action_short_vi(aid) == meta["short_vi"]
        assert format_action_devices_vi(aid) == meta["devices_vi"]


def test_controller_meta_vi_covers_all_6_controllers():
    expected = {
        "Double Q-Learning",
        "Single Q-Learning",
        "Fixed Setpoint (25°C Cool)",
        "Rule-Based Smart HVAC",
        "Conventional Thermostat",
        "Random Policy",
    }
    assert expected.issubset(set(CONTROLLER_META_VI.keys()))
    for k in expected:
        assert len(CONTROLLER_META_VI[k]["display_vi"]) > 5
        assert len(CONTROLLER_META_VI[k]["short_vi"]) > 4


def test_gui_vietnamese_instantiation_and_step():
    import tkinter as tk
    from app.live_simulation_gui import LiveSimulationApp

    root = tk.Tk()
    root.withdraw()
    try:
        app = LiveSimulationApp(root)
        assert "Phòng Điều Khiển" in root.title() or "Smart HVAC" in root.title()
        assert "device_cfg" in app.tree["columns"]
        for _ in range(3):
            app.step_forward_simulation()
        assert len(app.tree.get_children()) == 3
        dev_lbl_text = app.lbl_device_config.cget("text")
        assert "Thiết lập" in dev_lbl_text
    finally:
        root.destroy()


def test_pptx_minimum_font_size_20pt_and_pagination():
    import pptx

    pptx_path = next(
        (REPO_ROOT / "doc" / n for n in ("Nhom2_BaoCao.pptx", "BAO_CAO_THUYET_TRINH_FINAL.pptx") if (REPO_ROOT / "doc" / n).exists()),
        REPO_ROOT / "doc" / "Nhom2_BaoCao.pptx",
    )
    assert pptx_path.exists()
    prs = pptx.Presentation(str(pptx_path))
    n_slides = len(prs.slides)
    assert n_slides >= 26

    min_pt = 999.0
    all_text = []
    for slide in prs.slides:
        for shp in slide.shapes:
            if shp.has_text_frame:
                all_text.append(shp.text_frame.text)
                for p in shp.text_frame.paragraphs:
                    for r in p.runs:
                        if r.text.strip() and r.font.size:
                            min_pt = min(min_pt, round(r.font.size.pt, 1))
            if shp.has_table:
                for row in shp.table.rows:
                    for cell in row.cells:
                        all_text.append(cell.text)
                        for p in cell.text_frame.paragraphs:
                            for r in p.runs:
                                if r.text.strip() and r.font.size:
                                    min_pt = min(min_pt, round(r.font.size.pt, 1))
    assert min_pt >= 20.0, f"Found text < 20pt: {min_pt}pt"
    assert f"Trang {n_slides} / {n_slides}" in " ".join(all_text)


def test_docx_real_seq_captions_and_table_of_figures():
    import docx

    docx_path = next(
        (REPO_ROOT / "doc" / n for n in ("Nhom2_BaoCao.docx", "BAO_CAO_DO_AN_FINAL.docx") if (REPO_ROOT / "doc" / n).exists()),
        REPO_ROOT / "doc" / "Nhom2_BaoCao.docx",
    )
    assert docx_path.exists()
    doc = docx.Document(str(docx_path))

    # Word splits Unicode text across adjacent <w:instrText> runs on save; concatenate per paragraph
    instr_texts = []
    for p in doc.paragraphs:
        joined = "".join(el.text or "" for el in p._element.xpath(".//w:instrText"))
        if joined:
            instr_texts.append(joined)
        for fld in p._element.xpath(".//w:fldSimple"):
            instr_texts.append(fld.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}instr") or "")

    all_instr = "\n".join(instr_texts)
    assert "SEQ Hình" in all_instr
    assert "SEQ Bảng" in all_instr
    assert 'TOC \\h \\z \\c "Hình"' in all_instr
    assert 'TOC \\h \\z \\c "Bảng"' in all_instr
    tof_paras = [p for p in doc.paragraphs if p.style.name == "table of figures"]
    assert len(tof_paras) >= 20
