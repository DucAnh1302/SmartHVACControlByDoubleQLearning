"""Generate high-resolution visual assets grounded strictly in real CU-BEMS CSVs & cited literature."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "doc" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"]})


def build_cover_hero():
    """Clean futuristic Smart HVAC + Double Q-Learning visual diagram for Slide 1."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=220)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    # Outer Digital Twin Container
    outer = patches.FancyBboxPatch((0.3, 0.3), 9.4, 8.2, boxstyle="round,pad=0.15,rounding_size=0.35",
                                   fc="#1E293B", ec="#22D3EE", lw=2.2)
    ax.add_patch(outer)
    ax.text(5.0, 8.05, "DIGITAL TWIN • PHÒNG 50m² (140m³)", color="#22D3EE",
            fontsize=12.5, fontweight="bold", ha="center", va="center")

    # Top row: Environment state sensors
    sensors = [
        (0.7, 6.2, "#0EA5E9", "Nhiệt độ\n24–26°C"),
        (3.0, 6.2, "#38BDF8", "Độ ẩm\n40–65%"),
        (5.3, 6.2, "#10B981", "Khí CO2\n< 800 ppm"),
        (7.6, 6.2, "#F59E0B", "Bụi PM2.5\n<= 35 µg/m³"),
    ]
    for sx, sy, col, lbl in sensors:
        box = patches.FancyBboxPatch((sx, sy), 1.8, 1.15, boxstyle="round,pad=0.06,rounding_size=0.2",
                                     fc="#0F172A", ec=col, lw=1.8)
        ax.add_patch(box)
        ax.text(sx + 0.9, sy + 0.58, lbl, color="#F8FAFC", fontsize=9.8, fontweight="bold", ha="center", va="center")

    # Middle Brain: Double Q-Learning QA / QB + Safety Gateway
    brain = patches.FancyBboxPatch((1.0, 3.45), 8.0, 1.95, boxstyle="round,pad=0.08,rounding_size=0.25",
                                   fc="#0B4F6C", ec="#22D3EE", lw=2.2)
    ax.add_patch(brain)
    ax.text(5.0, 4.95, "BỘ NÃO AI: DOUBLE Q-LEARNING", color="#F8FAFC", fontsize=12.5, fontweight="bold", ha="center")
    qa = patches.FancyBboxPatch((1.35, 3.68), 2.35, 0.95, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#0F172A", ec="#34D399", lw=1.6)
    qb = patches.FancyBboxPatch((3.90, 3.68), 2.35, 0.95, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#0F172A", ec="#34D399", lw=1.6)
    sg = patches.FancyBboxPatch((6.45, 3.68), 2.25, 0.95, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#450A0A", ec="#F87171", lw=1.6)
    for b in [qa, qb, sg]:
        ax.add_patch(b)
    ax.text(2.52, 4.15, "Bảng Q_A\n(720 × 12)", color="#34D399", fontsize=9.8, fontweight="bold", ha="center", va="center")
    ax.text(5.07, 4.15, "Bảng Q_B\n(Cập nhật chéo)", color="#34D399", fontsize=9.8, fontweight="bold", ha="center", va="center")
    ax.text(7.57, 4.15, "Cổng An toàn\n(6 Luật cứng)", color="#FCA5A5", fontsize=9.8, fontweight="bold", ha="center", va="center")

    # Arrows sensor -> brain -> actuators
    for ax_x in [1.6, 3.9, 6.2, 8.5]:
        ax.annotate("", xy=(ax_x, 5.45), xytext=(ax_x, 6.12), arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.6))

    # Bottom: 7 Actuators & Window
    acts = [
        (0.55, 0.85, "Máy lạnh AC\n23–27°C"),
        (2.38, 0.85, "Quạt gió\n150–500 m³/h"),
        (4.21, 0.85, "Cửa sổ\nĐÓNG / MỞ"),
        (6.04, 0.85, "Lọc HEPA\n80–450 m³/h"),
        (7.87, 0.85, "Sưởi & Tạo/\nHút ẩm"),
    ]
    for ax_x, ay_y, albl in acts:
        ab = patches.FancyBboxPatch((ax_x, ay_y), 1.58, 1.45, boxstyle="round,pad=0.05,rounding_size=0.18",
                                    fc="#0F172A", ec="#10B981", lw=1.6)
        ax.add_patch(ab)
        ax.text(ax_x + 0.79, ay_y + 0.72, albl, color="#ECFDF5", fontsize=8.4, fontweight="bold", ha="center", va="center")
        ax.annotate("", xy=(ax_x + 0.79, 2.38), xytext=(ax_x + 0.79, 3.38),
                    arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.8))

    out = ASSETS_DIR / "hero_smart_hvac_cover.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    print(f"[+] Built cover hero -> {out}")


def build_hvac_evidence_visual():
    """Creates authentic evidence panel citing IEA, Harvard COGfx Study, and REAL CU-BEMS Floor 2 AC share."""
    df = pd.read_csv(REPO_ROOT / "data" / "raw" / "2018Floor2.csv")
    ac_cols = [c for c in df.columns if "AC" in c and "(kW)" in c]
    light_cols = [c for c in df.columns if "Light" in c and "(kW)" in c]
    plug_cols = [c for c in df.columns if "Plug" in c and "(kW)" in c]

    ac_kwh = float(df[ac_cols].sum().sum() / 60.0)
    light_kwh = float(df[light_cols].sum().sum() / 60.0)
    plug_kwh = float(df[plug_cols].sum().sum() / 60.0)
    tot = ac_kwh + light_kwh + plug_kwh

    fig = plt.figure(figsize=(12.2, 4.9), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.22)

    # Panel 1: REAL CU-BEMS 2018Floor2.csv Power Share Bar/Donut
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    shares = [ac_kwh / tot * 100.0, plug_kwh / tot * 100.0, light_kwh / tot * 100.0]
    labels = [f"Điều hòa AC\n({shares[0]:.1f}%)", f"Ổ cắm/TB\n({shares[1]:.1f}%)", f"Chiếu sáng\n({shares[2]:.1f}%)"]
    bars = ax1.bar(labels, shares, color=["#E11D48", "#3B82F6", "#F59E0B"], width=0.55)
    for b, v in zip(bars, shares):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11.5, color="#0F172A")
    ax1.set_ylim(0, 95)
    ax1.set_ylabel("% Tổng điện năng Tầng 2", fontsize=10.5, fontweight="bold")
    ax1.set_title("Đo thực tế trên CU-BEMS (2018Floor2.csv)\n264,960 phút đo vận hành văn phòng", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: IEA The Future of Cooling Evidence Card
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    rect2 = patches.FancyBboxPatch((0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0.02,rounding_size=0.06",
                                   fc="#FFFFFF", ec="#0891B2", lw=2.0)
    ax2.add_patch(rect2)
    ax2.text(0.5, 0.85, "BÁO CÁO QUỐC TẾ — IEA", ha="center", fontsize=11.5, fontweight="bold", color="#0891B2")
    ax2.text(0.5, 0.72, "The Future of Cooling\n(Cơ quan Năng lượng Quốc tế)", ha="center", fontsize=10.0, color="#475569")
    ax2.text(0.5, 0.47, ">= 50%", ha="center", fontsize=28, fontweight="bold", color="#E11D48")
    ax2.text(0.5, 0.24, "phụ tải đỉnh lưới điện tòa nhà\ntại các vùng nóng do HVAC gánh\n(> 2,900 TWh nhu cầu toàn cầu)",
             ha="center", va="center", fontsize=9.8, color="#0F172A")

    # Panel 3: Harvard COGfx Study Real Data Chart (CO2 vs Cognitive Score)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#FFFFFF")
    co2_levels = ["550 ppm\nThoáng", "945 ppm\nPhòng kín", "1,400 ppm\nNgạt CO2"]
    drops = [0, -15, -50]
    b3 = ax3.bar(co2_levels, drops, color=["#10B981", "#F59E0B", "#E11D48"], width=0.48)
    for b, d, txt in zip(b3, drops, ["Chuẩn", "-15%", "-50%"]):
        y_p = d - 4.8 if d < 0 else 2.2
        ax3.text(b.get_x() + b.get_width() / 2, y_p, txt, ha="center", fontweight="bold", fontsize=11.0, color="#0F172A")
    ax3.axhline(0, color="#0F172A", lw=1.0)
    ax3.set_ylim(-64, 14)
    ax3.tick_params(axis="x", labelsize=9.5)
    ax3.set_ylabel("Mức suy giảm nhận thức (%)", fontsize=10.0, fontweight="bold")
    ax3.set_title("Nghiên cứu COGfx — ĐH Harvard (EHP 2016)\nAllen et al.: Ảnh hưởng CO2 lên nhận thức", fontsize=10.8, fontweight="bold", color="#0F172A")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    out = ASSETS_DIR / "hvac_problem_evidence.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"[+] Built real evidence graphic (AC share={shares[0]:.1f}%) -> {out}")


def _draw_excel_sheet(ax, title: str, tab_names: list[str], active_idx: int,
                      headers: list[str], rows: list[list[str]], highlight_nulls: bool = False,
                      compact_tabs: bool = False):
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Excel window header green bar
    ax.add_patch(patches.Rectangle((0, 90), 100, 10, fc="#107C41", ec="#0B5A2F", lw=1.2))
    ax.text(2, 95, title, color="white", fontsize=10.5 if compact_tabs else 11.5, fontweight="bold", va="center")

    # Column letter header bar (A, B, C...)
    ncols = len(headers)
    col_w = 94.0 / ncols
    ax.add_patch(patches.Rectangle((0, 82), 6, 8, fc="#E2E8F0", ec="#CBD5E1"))
    ax.text(3, 86, "", ha="center", va="center", fontsize=9)
    for c_i in range(ncols):
        cx = 6 + c_i * col_w
        ax.add_patch(patches.Rectangle((cx, 82), col_w, 8, fc="#E2E8F0", ec="#94A3B8", lw=0.8))
        ax.text(cx + col_w / 2, 86, chr(65 + c_i), ha="center", va="center", fontsize=9.2, fontweight="bold", color="#334155")

    # Data headers (Row 1) + Rows (Row 2..N+1)
    all_rows = [headers] + rows
    nrows = len(all_rows)
    row_h = 68.0 / nrows
    for r_i, r_vals in enumerate(all_rows):
        ry = 82 - (r_i + 1) * row_h
        ax.add_patch(patches.Rectangle((0, ry), 6, row_h, fc="#E2E8F0", ec="#94A3B8", lw=0.8))
        ax.text(3, ry + row_h / 2, str(r_i + 1), ha="center", va="center", fontsize=8.5, fontweight="bold", color="#475569")
        for c_i, val in enumerate(r_vals):
            cx = 6 + c_i * col_w
            is_hdr = (r_i == 0)
            is_null = highlight_nulls and (val in ["NaN", "null", "TRỐNG"])
            fc = "#DCFCE7" if is_hdr else ("#FEE2E2" if is_null else ("#FFFFFF" if r_i % 2 == 1 else "#F8FAFC"))
            ec = "#DC2626" if is_null else "#CBD5E1"
            ax.add_patch(patches.Rectangle((cx, ry), col_w, row_h, fc=fc, ec=ec, lw=1.2 if is_null else 0.7))
            ax.text(cx + col_w / 2, ry + row_h / 2, str(val), ha="center", va="center",
                    fontsize=8.2 if is_hdr else 8.0,
                    fontweight="bold" if (is_hdr or is_null) else "normal",
                    color="#14532D" if is_hdr else ("#B91C1C" if is_null else "#0F172A"))

    # Bottom Excel Sheet Tabs scaled cleanly for full or split subplots
    tx = 1.0
    n_tabs = len(tab_names)
    avail = 96.0 / max(1, n_tabs)
    for ti, tname in enumerate(tab_names):
        tw = min(avail - 1.5, max(22.0 if compact_tabs else 15.0, len(tname) * (2.6 if compact_tabs else 1.25)))
        if compact_tabs and n_tabs == 1:
            tw = 55.0
        elif compact_tabs and n_tabs == 2:
            tw = 44.0
        is_act = (ti == active_idx)
        ax.add_patch(patches.Rectangle((tx, 2), tw, 9.5, fc="#FFFFFF" if is_act else "#E2E8F0",
                                       ec="#107C41" if is_act else "#94A3B8", lw=1.8 if is_act else 0.8))
        ax.text(tx + tw / 2, 6.7, tname, ha="center", va="center", fontsize=8.0 if compact_tabs else 8.5,
                fontweight="bold" if is_act else "normal", color="#107C41" if is_act else "#475569")
        tx += tw + 1.8


def build_raw_excel_preview():
    """Creates authentic Excel-style preview of raw 2018Floor2.csv (including real 1-min rows & real NaN rows)."""
    raw = pd.read_csv(REPO_ROOT / "data" / "raw" / "2018Floor2.csv", usecols=[
        "Date", "z1_AC1(kW)", "z1_Light(kW)", "z1_Plug(kW)", "z1_S1(degC)", "z1_S1(RH%)"
    ])
    # Pick 4 initial rows (0..3) and a slice with real missing sensor NaNs to show why cleaning is needed
    nan_idx = raw[raw["z1_S1(degC)"].isna()].index
    sample_rows = []
    for idx in [0, 1, 2, 3]:
        r = raw.iloc[idx]
        sample_rows.append([
            str(r["Date"])[5:16], f"{r['z1_AC1(kW)']:.2f}", f"{r['z1_Light(kW)']:.2f}",
            f"{r['z1_Plug(kW)']:.2f}", f"{r['z1_S1(degC)']:.2f}", f"{r['z1_S1(RH%)']:.2f}"
        ])
    # Peak daytime row (~09:00 on weekday July 2)
    wk_idx = raw[raw["Date"].str.startswith("2018-07-02 09:0")].index[:2]
    for idx in wk_idx:
        r = raw.iloc[idx]
        sample_rows.append([
            str(r["Date"])[5:16],
            f"{r['z1_AC1(kW)']:.2f}" if pd.notna(r["z1_AC1(kW)"]) else "NaN",
            f"{r['z1_Light(kW)']:.2f}" if pd.notna(r["z1_Light(kW)"]) else "NaN",
            f"{r['z1_Plug(kW)']:.2f}" if pd.notna(r["z1_Plug(kW)"]) else "NaN",
            f"{r['z1_S1(degC)']:.2f}" if pd.notna(r["z1_S1(degC)"]) else "NaN",
            f"{r['z1_S1(RH%)']:.2f}" if pd.notna(r["z1_S1(RH%)"]) else "NaN",
        ])
    # Real NaN row slice
    if len(nan_idx) > 0:
        for idx in nan_idx[:2]:
            r = raw.iloc[idx]
            sample_rows.append([
                str(r["Date"])[5:16],
                f"{r['z1_AC1(kW)']:.2f}" if pd.notna(r["z1_AC1(kW)"]) else "NaN",
                f"{r['z1_Light(kW)']:.2f}" if pd.notna(r["z1_Light(kW)"]) else "NaN",
                f"{r['z1_Plug(kW)']:.2f}" if pd.notna(r["z1_Plug(kW)"]) else "NaN",
                "NaN" if pd.isna(r["z1_S1(degC)"]) else f"{r['z1_S1(degC)']:.2f}",
                "NaN" if pd.isna(r["z1_S1(RH%)"]) else f"{r['z1_S1(RH%)']:.2f}",
            ])

    fig, ax = plt.subplots(figsize=(11.8, 4.8), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    _draw_excel_sheet(
        ax,
        "EXCEL VIEW — BỘ DỮ LIỆU GỐC THÔ: data/raw/2018Floor2.csv (264,960 dòng từng phút × 37 cột)",
        ["2018Floor1.csv", "2018Floor2.csv", "2018Floor3..7.csv", "2019Floor1..7.csv"],
        1,
        ["Date (Mỗi 1 phút)", "z1_AC1 (kW)", "z1_Light (kW)", "z1_Plug (kW)", "z1_S1 (°C)", "z1_S1 (RH%)"],
        sample_rows,
        highlight_nulls=True,
    )
    out = ASSETS_DIR / "cubems_raw_excel_preview.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"[+] Built raw Excel preview -> {out}")


def build_floor2_transform_visual():
    """Side-by-side transformation visual: Raw 2018Floor2.csv -> Processed 2018_F2_Z1_1h.csv."""
    raw = pd.read_csv(REPO_ROOT / "data" / "raw" / "2018Floor2.csv", usecols=[
        "Date", "z1_AC1(kW)", "z1_Light(kW)", "z1_Plug(kW)", "z1_S1(degC)", "z1_S1(RH%)"
    ])
    proc = pd.read_csv(REPO_ROOT / "data" / "processed" / "2018_F2_Z1_1h.csv")
    # Weekday morning slice 2018-07-02 07:00 -> 13:00
    sub_raw = raw[raw["Date"].str.startswith("2018-07-02 08:0")].head(6)
    raw_rows = [
        [str(r["Date"])[11:19], f"{r['z1_AC1(kW)']:.2f}", f"{r['z1_Plug(kW)']:.2f}", f"{r['z1_S1(degC)']:.2f}"]
        for _, r in sub_raw.iterrows()
    ]

    sub_proc = proc[proc["datetime"].str.startswith("2018-07-02")].iloc[7:13]
    proc_rows = [
        [str(r["datetime"])[5:16], f"{r['hvac_kw']:.2f}", f"{r['temp_c']:.2f}",
         f"{r['internal_heat_scaled_kw']:.2f}", f"{int(r['occupancy_est'])} ({'Có người' if r['occupancy_est'] else 'Trống'})"]
        for _, r in sub_proc.iterrows()
    ]

    fig = plt.figure(figsize=(12.4, 4.7), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.55, 1.25], wspace=0.06)

    axL = fig.add_subplot(gs[0, 0])
    _draw_excel_sheet(
        axL,
        "1. GỐC: 2018Floor2.csv (1 phút)",
        ["2018Floor2 (264k dòng)"], 0,
        ["Giờ đo (1p)", "z1_AC1", "z1_Plug", "z1_T (°C)"],
        raw_rows, highlight_nulls=True, compact_tabs=True
    )

    axM = fig.add_subplot(gs[0, 1])
    axM.axis("off")
    axM.set_xlim(0, 10)
    axM.set_ylim(0, 10)
    steps = [
        (7.7, "#0891B2", "1. Lọc Zone 1\n(Tầng 2)"),
        (5.5, "#4F46E5", "2. Gom 60 dòng\n-> TB 1 giờ"),
        (3.3, "#D97706", "3. Nội suy đứt\nquãng <= 3h"),
        (1.1, "#059669", "4. Quy 50m² &\nSuy người (96.8%)"),
    ]
    for sy, scol, stxt in steps:
        box = patches.FancyBboxPatch((0.6, sy), 8.8, 1.55, boxstyle="round,pad=0.1,rounding_size=0.35", fc=scol, ec="none")
        axM.add_patch(box)
        axM.text(5.0, sy + 0.77, stxt, color="white", fontsize=8.8, fontweight="bold", ha="center", va="center")

    axR = fig.add_subplot(gs[0, 2])
    _draw_excel_sheet(
        axR,
        "2. SAU XỬ LÝ: 2018_F2_Z1_1h.csv (1,767 giờ)",
        ["2018_F2_Z1_1h.csv", "2019_F2_Z1_1h.csv"], 0,
        ["Giờ (1h)", "hvac_kW", "temp_c", "Tải 50m²", "Số người"],
        proc_rows, highlight_nulls=False, compact_tabs=True
    )

    out = ASSETS_DIR / "cubems_floor2_transform.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close(fig)
    print(f"[+] Built 2018Floor2 -> Processed transform -> {out}")


def main():
    build_cover_hero()
    build_hvac_evidence_visual()
    build_raw_excel_preview()
    build_floor2_transform_visual()


if __name__ == "__main__":
    main()
