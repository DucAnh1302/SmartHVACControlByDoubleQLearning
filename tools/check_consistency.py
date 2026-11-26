"""
Global Consistency Audit Checker (Phase 4-8).
Verifies cross-artifact integrity between ground-truth code/manifest/CSVs and
all Markdown docs, Excel workbook, 18 Figures, Word report (.docx), and PowerPoint (.pptx).
"""

import hashlib
import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_consistency(check_docx: bool = False, check_pptx: bool = False):
    print("=" * 70)
    print("  SMART HVAC GLOBAL CONSISTENCY CHECK")
    print("=" * 70)

    # 1. Load ground truth sources
    meta_path = REPO_ROOT / "config" / "project_meta.yaml"
    manifest_path = REPO_ROOT / "result" / "final" / "manifest.json"
    if not manifest_path.exists():
        print("[!] CRITICAL: manifest.json missing. Run Phase 3 benchmark first.")
        return False, ["Missing manifest.json"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    instr = meta.get("instructor", {})
    sup_title = instr.get("title", "TS.").strip()
    sup_name = instr.get("name", "Nguyễn Thanh Long").strip()
    sup_full = f"{sup_title} {sup_name}"

    p_val_t = manifest["statistical_tests"]["paired_t"]["p_value"]
    t_stat = manifest["statistical_tests"]["paired_t"]["statistic"]
    branch = manifest["interpretation_branch"]
    dql_testA = manifest["controllers"]["Double Q-Learning"]["test-A"]
    ql_testA = manifest["controllers"]["Single Q-Learning"]["test-A"]
    dql_e = float(dql_testA["energy_kwh_per_day"].split("±")[0].strip())
    ql_e = float(ql_testA["energy_kwh_per_day"].split("±")[0].strip())
    dql_r = float(dql_testA["mean_reward"].split("±")[0].strip())
    ql_r = float(ql_testA["mean_reward"].split("±")[0].strip())

    print("[*] Ground Truth Metadata:")
    print(f"    - GVHD: {sup_full}")
    print(f"    - DQL Test-A: Reward={dql_r:.2f}, Energy={dql_e:.2f} kWh/day")
    print(f"    - QL  Test-A: Reward={ql_r:.2f}, Energy={ql_e:.2f} kWh/day")
    print(f"    - Paired t-test: t={t_stat:.4f}, p={p_val_t:.4f}")
    print(f"    - Interpretation Branch: {branch}")

    discrepancies = []

    banned_supervisor = ["Nguyễn Thành Long"]
    if sup_title == "TS.":
        banned_supervisor.append("ThS. Nguyễn Thanh Long")

    # 2. Check CSV Design Tables vs Live Code Constants
    tbl_dir = REPO_ROOT / "result" / "final" / "tables"
    required_csvs = [
        "state_table.csv",
        "action_table.csv",
        "reward_table.csv",
        "safety_table.csv",
        "hyperparams_table.csv",
    ]
    for cname in required_csvs:
        cp = tbl_dir / cname
        if not cp.exists():
            discrepancies.append(f"Missing CSV table: {cp}")

    if (tbl_dir / "reward_table.csv").exists():
        rw_txt = (tbl_dir / "reward_table.csv").read_text(encoding="utf-8")
        if "0.12" not in rw_txt or "0.35" not in rw_txt:
            discrepancies.append("reward_table.csv missing real code weights w_energy=0.12 / w_switch=0.35")
        if ",-5.00" in rw_txt or ",0.6," in rw_txt:
            discrepancies.append("reward_table.csv contains stale fabricated reward weights")

    if (tbl_dir / "safety_table.csv").exists():
        sg_txt = (tbl_dir / "safety_table.csv").read_text(encoding="utf-8")
        if "29.5°C" not in sg_txt or "1600" not in sg_txt or "18.0°C" not in sg_txt:
            discrepancies.append("safety_table.csv does not match MultiDeviceSafetyConfig (29.5C, 18.0C, 1600 ppm)")
        if "32.0°C" in sg_txt or "16.0°C" in sg_txt or "1800.0 ppm" in sg_txt:
            discrepancies.append("safety_table.csv contains stale safety thresholds")

    if (tbl_dir / "state_table.csv").exists():
        st_txt = (tbl_dir / "state_table.csv").read_text(encoding="utf-8")
        if "96.8%" not in st_txt or "0.50 kW" not in st_txt:
            discrepancies.append("state_table.csv missing 0.50 kW load threshold / 96.8% match description")

    # 3. Check 18 PNG figures and Excel workbook (4 sheets)
    fig_dir = REPO_ROOT / "result" / "final" / "figures"
    pngs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if len(pngs) < 18:
        discrepancies.append(f"Expected 18 figures in result/final/figures, found {len(pngs)}")

    xlsx_path = REPO_ROOT / "doc" / "THIET_KE_PHAN_THUONG_VA_HANH_DONG_HVAC.xlsx"
    if not xlsx_path.exists():
        discrepancies.append("Missing Excel workbook THIET_KE_PHAN_THUONG_VA_HANH_DONG_HVAC.xlsx")
    else:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        if len(wb.sheetnames) != 4:
            discrepancies.append(f"Excel workbook expected 4 sheets, found {len(wb.sheetnames)}")

    # 4. Check Markdown Documents
    docs_to_check = [
        REPO_ROOT / "docs" / "STATE.md",
        REPO_ROOT / "docs" / "HANDOFF.md",
        REPO_ROOT / "docs" / "EXPERIMENT_PROTOCOL.md",
        REPO_ROOT / "README.md",
    ]
    for dp in docs_to_check:
        if dp.exists():
            text = dp.read_text(encoding="utf-8")
            for b in banned_supervisor:
                if b in text:
                    discrepancies.append(f"{dp.name} contains banned supervisor string '{b}'")
            for old in ["68 bài", "68/68", "−3,910", "48.03", "42.5%", "49.6%"]:
                if old in text:
                    discrepancies.append(f"{dp.name} contains legacy stale token '{old}'")

    # 5. Check Word .docx if requested
    if check_docx:
        docx_path = next(
            (REPO_ROOT / "doc" / n for n in ("Nhom2_BaoCao.docx", "BAO_CAO_DO_AN_FINAL.docx") if (REPO_ROOT / "doc" / n).exists()),
            REPO_ROOT / "doc" / "Nhom2_BaoCao.docx",
        )
        if not docx_path.exists():
            discrepancies.append("Missing doc/Nhom2_BaoCao.docx (or BAO_CAO_DO_AN_FINAL.docx)")
        else:
            try:
                import docx
                doc = docx.Document(str(docx_path))
                doc_text = " ".join(
                    [p.text for p in doc.paragraphs]
                    + [c.text for t in doc.tables for r in t.rows for c in r.cells]
                )

                for b in banned_supervisor:
                    if b in doc_text:
                        discrepancies.append(f"Word document contains wrong supervisor title/name: '{b}'")
                if sup_full not in doc_text:
                    discrepancies.append(f"Word document missing expected supervisor '{sup_full}'")

                forbidden_docx = [
                    "−3,910", "48.03", "42.5%", "49.6%",
                    "Bảng: Bảng", "Hình: Hình",
                    "Chintanapakdee", "12345", "D. L. F. O.",
                    "HVACDigitalTwinMultiDeviceEnv", "double_q_learning_agent.py",
                    "test_reward.py", "68 bài", "68/68", "68 Tests", "68 TEST",
                    "100% các tiêu chí", "thổi phồng giá trị Q ước lượng lên cao hơn giá trị thực tế tới 28%",
                    "ở mức tuyệt đối 0.0%, trong khi Thermostat", "14.2%", "4.2%, thấp hơn",
                ]
                for tok in forbidden_docx:
                    if tok in doc_text:
                        discrepancies.append(f"Word document contains forbidden/stale token: '{tok}'")

                if f"{dql_e:.2f}" not in doc_text:
                    discrepancies.append(f"Word document missing current Test-A DQL energy mean {dql_e:.2f}")
                if f"{ql_e:.2f}" not in doc_text:
                    discrepancies.append(f"Word document missing current Test-A QL energy mean {ql_e:.2f}")
                if f"{t_stat:.4f}" not in doc_text:
                    discrepancies.append(f"Word document missing current paired t-statistic {t_stat:.4f}")

                # Check TOC/Footer PAGE field XML
                has_page_fld = any("PAGE" in sec.footer._element.xml for sec in doc.sections)
                if not has_page_fld:
                    discrepancies.append("Word document footer missing dynamic PAGE field")

                # Check embedded image hashes match current result/final/figures/*.png (or cover logos)
                fig_hashes = {_file_md5(p): p.name for p in pngs}
                embedded_blobs = [part.blob for rel in doc.part.rels.values() if "image" in rel.reltype for part in [rel.target_part]]
                matched_figs = sum(1 for b in embedded_blobs if hashlib.md5(b).hexdigest() in fig_hashes)
                if matched_figs < 6:
                    discrepancies.append(
                        f"Word document embedded charts appear stale or mismatched (only {matched_figs} match result/final/figures/*.png)"
                    )

                # Also audit SO_TAY_DOC_HINH_ANH_BIEU_DO_HVAC.docx
                hb_path = REPO_ROOT / "doc" / "SO_TAY_DOC_HINH_ANH_BIEU_DO_HVAC.docx"
                if hb_path.exists():
                    hb_doc = docx.Document(str(hb_path))
                    hb_txt = " ".join(
                        [p.text for p in hb_doc.paragraphs]
                        + [c.text for t in hb_doc.tables for r in t.rows for c in r.cells]
                    )
                    for b in banned_supervisor:
                        if b in hb_txt:
                            discrepancies.append(f"Handbook contains banned supervisor string: '{b}'")
                    for tok in ["0.00032", "-6000", "34.2%", "31.8%", "multi_device_training_curves"]:
                        if tok in hb_txt:
                            discrepancies.append(f"Handbook contains legacy stale token: '{tok}'")
                    if f"{dql_e:.2f}" not in hb_txt or f"{t_stat:.4f}" not in hb_txt:
                        discrepancies.append("Handbook missing current manifest energy or paired t-stat")
            except Exception as e:
                discrepancies.append(f"Error reading Word document: {e}")

    # 6. Check PowerPoint .pptx if requested
    if check_pptx:
        pptx_path = next(
            (REPO_ROOT / "doc" / n for n in ("Nhom2_BaoCao.pptx", "BAO_CAO_THUYET_TRINH_FINAL.pptx") if (REPO_ROOT / "doc" / n).exists()),
            REPO_ROOT / "doc" / "Nhom2_BaoCao.pptx",
        )
        if not pptx_path.exists():
            discrepancies.append("Missing doc/Nhom2_BaoCao.pptx (or BAO_CAO_THUYET_TRINH_FINAL.pptx)")
        else:
            try:
                import pptx
                prs = pptx.Presentation(str(pptx_path))
                all_slide_texts = []
                min_font_pt = 999.0
                for si, slide in enumerate(prs.slides, 1):
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            all_slide_texts.append(shape.text_frame.text)
                            for para in shape.text_frame.paragraphs:
                                for run in para.runs:
                                    if run.text.strip() and run.font.size:
                                        pt = round(run.font.size.pt, 1)
                                        if pt < min_font_pt:
                                            min_font_pt = pt
                        if shape.has_table:
                            for row in shape.table.rows:
                                for cell in row.cells:
                                    all_slide_texts.append(cell.text)
                                    for para in cell.text_frame.paragraphs:
                                        for run in para.runs:
                                            if run.text.strip() and run.font.size:
                                                pt = round(run.font.size.pt, 1)
                                                if pt < min_font_pt:
                                                    min_font_pt = pt
                    if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                        all_slide_texts.append(slide.notes_slide.notes_text_frame.text)

                ppt_text = " ".join(all_slide_texts)

                for b in banned_supervisor:
                    if b in ppt_text:
                        discrepancies.append(f"PowerPoint contains wrong supervisor title/name: '{b}'")
                if sup_full not in ppt_text:
                    discrepancies.append(f"PowerPoint missing expected supervisor '{sup_full}'")

                forbidden_pptx = [
                    "2R-2C", "68 bài", "−3,910", "48.03", "42.5%", "49.6%",
                    "Nghỉ nén >= 3 bước", ">32°C; Tắt <16°C", "IAQ..", "Bias..", "toàn..",
                ]
                for tok in forbidden_pptx:
                    if tok in ppt_text:
                        discrepancies.append(f"PowerPoint contains forbidden/stale token: '{tok}'")

                total_slides = len(prs.slides)
                expected_last_badge = f"Trang {total_slides} / {total_slides}"
                if expected_last_badge not in ppt_text:
                    discrepancies.append(f"PowerPoint closing Slide missing pagination badge '{expected_last_badge}'")

                if min_font_pt < 20.0:
                    discrepancies.append(f"PowerPoint contains text smaller than 20pt ({min_font_pt:.1f}pt)")

                if f"{dql_e:.2f}" not in ppt_text or f"{ql_e:.2f}" not in ppt_text:
                    discrepancies.append("PowerPoint missing current Test-A DQL/QL energy numbers")
            except Exception as e:
                discrepancies.append(f"Error reading PowerPoint presentation: {e}")

    print("\n" + "=" * 70)
    if discrepancies:
        print(f"CONSISTENCY CHECK RESULT: DISCREPANCIES DETECTED ({len(discrepancies)})")
        for d in discrepancies:
            print(f"  [!] {d}")
        return False, discrepancies
    else:
        print("CONSISTENCY CHECK RESULT: PASS (All verified targets match ground truth)")
        return True, []


def main():
    check_docx = "--check-docx" in sys.argv
    check_pptx = "--check-pptx" in sys.argv
    success, _ = check_consistency(check_docx=check_docx, check_pptx=check_pptx)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
